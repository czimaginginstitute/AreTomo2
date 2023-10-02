#include "CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Util/Util_Time.h>
#include <nvToolsExt.h>

using namespace DfCorr;
using namespace DfCorr::Correct;

// padded sizeX, sizeY, number of frames, number of patches
static __device__ __constant__ int giSizes[4];

static __global__ void mGCorrect3D
(	float* gfPadFrmIn,
	int iFrame,
	float* gfPatCenters,
	float* gfPatShifts,
	float* gfPadFrmOut
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= giSizes[1]) return;
	int iOut = y * giSizes[0] + blockIdx.x;
	//-------------------------------------
	float afXYZ[2] = {0.0f};
	float fSx = 0.0f, fSy = 0.0f, fW = 0.0f;
	for(int p=0; p<giSizes[3]; p++)
	{	int k =  p * 2;
		afXYZ[0] = (blockIdx.x - gfPatCenters[k]) / gridDim.x;
		afXYZ[1] = (y - gfPatCenters[k+1]) / giSizes[1];
		if(fabsf(afXYZ[0]) > 0.5f) continue;
		if(fabsf(afXYZ[1]) > 0.5f) continue;
		afXYZ[0] = expf(-36.0f * (afXYZ[0] * afXYZ[0]
		   + afXYZ[1] * afXYZ[1]));
		k = p * giSizes[2] * 2 + iFrame;
		fSx += gfPatShifts[k] * afXYZ[0];
		fSy += gfPatShifts[k+giSizes[2]] * afXYZ[0];
		fW += afXYZ[0];
	}
	int x = (int)(blockIdx.x + fSx / fW);
	y = (int)(y + fSy / fW);
	//----------------------
	if(x < 0 || y < 0 || x >= gridDim.x || y >= giSizes[1])
	{	x = (x < 0) ? -x : x;
		y = (y < 0) ? -y : y;
		x = (811 * x) % gridDim.x;
		y = (811 * y) % giSizes[1];
	}
	//---------------------------------
	gfPadFrmOut[iOut] = gfPadFrmIn[y * giSizes[0] + x];
}

Align::CPatchShifts* GCorrectPatchShift::m_pPatchShifts = 0L;

GCorrectPatchShift::GCorrectPatchShift(void)
{
	m_aBlockDim.x = 1;
	m_aBlockDim.y = 64;
}

GCorrectPatchShift::~GCorrectPatchShift(void)
{
}

void GCorrectPatchShift::DoIt
(	Align::CPatchShifts* pPatchShifts,
	MrcUtil::CAlnSums* pAlnSums,
	MrcUtil::CMrcStack* pAlnStack
)
{	nvtxRangePushA("GCorrectPatchShift::DoIt");
	Util_Time utilTime; utilTime.Measure();
	//-------------------------------------
	m_pPatchShifts = pPatchShifts;
	m_pFullShift = pPatchShifts->m_pFullShift;
	m_pAlnSums = pAlnSums;
	m_pAlnStack = pAlnStack;
	m_aiOutCmpSize[0] = pAlnSums->m_aiStkSize[0] / 2 + 1;
	m_aiOutCmpSize[1] = pAlnSums->m_aiStkSize[1];
	m_aiOutPadSize[0] = m_aiOutCmpSize[0] * 2;
	m_aiOutPadSize[1] = m_aiOutCmpSize[1];
	//------------------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(EBuffer::frm);
	m_aiInCmpSize[0] = pFrmBuffer->m_aiCmpSize[0];
	m_aiInCmpSize[1] = pFrmBuffer->m_aiCmpSize[1];
	m_aiInPadSize[0] = m_aiInCmpSize[0] * 2;
	m_aiInPadSize[1] = m_aiInCmpSize[1];
	//----------------------------------
	int iNumGpus = pBufferPool->m_iNumGpus;
	GCorrectPatchShift* pThreads = new GCorrectPatchShift[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(i);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	//-------------------
	mSumPartialSums();
	mCorrectMag();
	mUnpadSums();
	//-----------
	float fSecs = utilTime.GetElapsedSeconds();
	printf("Correction of local motion: %f sec\n\n", fSecs); 
	nvtxRangePop();
}

void GCorrectPatchShift::Run(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	this->Start();
}

void GCorrectPatchShift::ThreadMain(void)
{
	CCorrectFullShift::mInit();
	//-------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	m_pForwardFFT = pBufferPool->GetForwardFFT(m_iNthGpu);
	m_pForwardFFT->CreateForwardPlan(m_aiInPadSize, true);
	//----------------------------------------------------
	int aiSizes[] = 
	{ m_aiInPadSize[0], m_aiInPadSize[1], 
	  m_pPatchShifts->m_aiFullSize[2],
	  m_pPatchShifts->m_iNumPatches
	};
	cudaMemcpyToSymbol(giSizes, aiSizes, sizeof(aiSizes));
	//----------------------------------------------------
	int iBytes = m_pPatchShifts->m_iNumPatches * 2 * sizeof(float);
	cudaMalloc(&m_gfPatCenters, iBytes);
	m_pPatchShifts->CopyCentersToGpu(m_gfPatCenters);
	//-----------------------------------------------
	iBytes = m_pPatchShifts->m_iNumPatches 
	   * m_pPatchShifts->m_aiFullSize[2] * 2 * sizeof(float);
	cudaMalloc(&m_gfPatShifts, iBytes);
	m_pPatchShifts->CopyShiftsToGpu(m_gfPatShifts);
	//---------------------------------------------
	m_aGridDim.x = (m_aiInCmpSize[0] - 1) * 2;
	m_aGridDim.y = (m_aiInCmpSize[1] + m_aBlockDim.y - 1) / m_aBlockDim.y;
	//--------------------------------------------------------------------
	mCorrectCpuFrames();
	mCorrectGpuFrames();
	//------------------
	CCorrectFullShift::Wait();
	cudaFree(m_gfPatCenters);
	cudaFree(m_gfPatShifts);
}

void GCorrectPatchShift::mCorrectCpuFrames(void)
{
	int iCount = 0;
	int iStartFrm = m_pFrmBuffer->GetStartFrame(m_iNthGpu);
	int iNumFrames = m_pFrmBuffer->GetNumFrames(m_iNthGpu);
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	cufftComplex* pCmpFrm = 0L;
	cufftComplex* gCmpBuf = m_pTmpBuffer->GetFrame(m_iNthGpu, 0); 
	cufftComplex* gCmpAln = m_pTmpBuffer->GetFrame(m_iNthGpu, 1);
	//-----------------------------------------------------------
	for(int i=0; i<iNumFrames; i++)
	{	if(m_pFrmBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		pCmpFrm = m_pFrmBuffer->GetFrame(m_iNthGpu, i);
		//---------------------------------------------
		m_iAbsFrm = iStartFrm + i;
		int iStream = iCount % 2;
		//-----------------------
		if(iStream == 1) cudaStreamSynchronize(m_aStreams[0]);
		cudaMemcpyAsync(gCmpBuf, pCmpFrm, tBytes, 
			cudaMemcpyDefault, m_aStreams[iStream]);
		if(iStream == 1) cudaStreamSynchronize(m_aStreams[1]);
		//----------------------------------------------------
		mAlignFrame(gCmpBuf);
		mGenSums(gCmpAln);
		iCount += 1;	
	}
}

void GCorrectPatchShift::mCorrectGpuFrames(void)
{
	int iStartFrm = m_pFrmBuffer->GetStartFrame(m_iNthGpu);
	int iNumFrames = m_pFrmBuffer->GetNumFrames(m_iNthGpu);
	cufftComplex* gCmpAln = m_pTmpBuffer->GetFrame(m_iNthGpu, 1);
	for(int i=0; i<iNumFrames; i++)
	{	if(!m_pFrmBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		//---------------------------------------------------
		m_iAbsFrm = iStartFrm + i;
		cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(m_iNthGpu, i);
		mAlignFrame(gCmpFrm);
		mGenSums(gCmpAln);	
	}
}

void GCorrectPatchShift::mAlignFrame(cufftComplex* gCmpFrm)
{
	float* gfPadFrm = reinterpret_cast<float*>(gCmpFrm);
	//--------------------------------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	float* gfPadAln = (float*)m_pTmpBuffer->GetFrame(m_iNthGpu, 1);
	//-------------------------------------------------------------
	mGCorrect3D<<<m_aGridDim, m_aBlockDim, 0, m_aStreams[0]>>>(gfPadFrm,
	   m_iAbsFrm, m_gfPatCenters, m_gfPatShifts, gfPadAln);
	//-----------------------------------------------------
	bool bNorm = true;
	m_pForwardFFT->Forward(gfPadAln, bNorm, m_aStreams[0]);
}

void GCorrectPatchShift::mMotionDecon(cufftComplex* gCmpFrm)
{	
}
