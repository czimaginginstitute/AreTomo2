#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace Recon;

static MrcUtil::CTomoStack* s_pTomoStack = 0L;
static MrcUtil::CAlignParam* s_pAlignParam = 0L;
static int s_iStartTilt = 0;
static int s_iNumTilts = 0;
static MrcUtil::CTomoStack* s_pVolStack = 0L;
static int s_iNumIters = 1;
static int s_iNumSubsets = 1;

MrcUtil::CTomoStack* CDoSartRecon::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	int iStartTilt,
	int iNumTilts,
	int iVolZ,
	int iIterations,
	int iNumSubsets,
	int* piGpuIDs,
	int iNumGpus
)
{	s_pTomoStack = pTomoStack;
	s_pAlignParam = pAlignParam;
	s_iStartTilt = iStartTilt;
	s_iNumTilts = iNumTilts;
	s_iNumIters = iIterations;
	s_iNumSubsets = iNumSubsets;
	//--------------------------
	int aiVolSize[3] = {1, iVolZ, pTomoStack->m_aiStkSize[1]};
	//aiVolSize[0] = (pTomoStack->m_aiStkSize[0] + 16) / 2 * 2;
	aiVolSize[0] = pTomoStack->m_aiStkSize[0] / 2 * 2;
	s_pVolStack = new MrcUtil::CTomoStack;
	s_pVolStack->Create(aiVolSize);
	//-----------------
	Util::CNextItem nextItem;
	nextItem.Create(pTomoStack->m_aiStkSize[1]);
	//-----------------
	CDoSartRecon* pThreads = new CDoSartRecon[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	printf("SART reconstruction completed.\n\n");
	//-------------------------------------------
	MrcUtil::CTomoStack* pVolStack = s_pVolStack;
	s_pVolStack = 0L;
	return pVolStack;		
}

CDoSartRecon::CDoSartRecon(void)
{
}

CDoSartRecon::~CDoSartRecon(void)
{
	this->Clean();
}

void CDoSartRecon::Clean(void)
{
	CDoBaseRecon::Clean();
	m_aTomoSart.Clean();
}

void CDoSartRecon::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	int iPadX = (s_pTomoStack->m_aiStkSize[0] / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iPadX * s_pTomoStack->m_aiStkSize[2];
	cudaMalloc(&m_gfPadSinogram, tBytes);
	cudaMallocHost(&m_pfPadSinogram, tBytes);	
	//---------------------------------------
	tBytes = s_pVolStack->GetPixels() * sizeof(float); 
	cudaMalloc(&m_gfVolXZ, tBytes);
	cudaMemset(m_gfVolXZ, 0, tBytes);
	cudaMallocHost(&m_pfVolXZ, tBytes);
	//---------------------------------
	m_aTomoSart.Setup
	( s_pVolStack->m_aiStkSize[0], s_pVolStack->m_aiStkSize[1],
	  s_iNumSubsets, s_iNumIters, s_pTomoStack, s_pAlignParam,
	  s_iStartTilt, s_iNumTilts
	);
	//-------------------------
	cudaStreamCreate(&m_stream);
	cudaEventCreate(&m_eventSino);
	//----------------------------
	int iLastY = -1;
	while(true)
	{	int iY = m_pNextItem->GetNext();
		if(iY < 0) break;
		//---------------
		if(iY % 101 == 0)
		{	int iLeft = s_pTomoStack->m_aiStkSize[1] - 1 - iY;
			printf("...... reconstruct slice %4d, "
			   "%4d slices left\n", iY+1, iLeft);
		}
		//-------------------------------------------
		mExtractSinogram(iY);
		mGetReconResult(iLastY);
		mReconstruct(iY);
		iLastY = iY;
	}
	cudaStreamSynchronize(m_stream);
	mGetReconResult(iLastY);
	//----------------------
	cudaStreamDestroy(m_stream);
	cudaEventDestroy(m_eventSino);
}

void CDoSartRecon::mExtractSinogram(int iY)
{
	int iProjX = s_pTomoStack->m_aiStkSize[0];
	int iPadX = (iProjX / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iProjX;
	for(int i=0; i<s_pTomoStack->m_aiStkSize[2]; i++)
	{	float* pfProj = s_pTomoStack->GetFrame(i);
		float* pfSrc = pfProj + iY * iProjX;
		float* pfDst = m_pfPadSinogram + i * iPadX;
		memcpy(pfDst, pfSrc, tBytes);
	}
	//-----------------------------------
	cudaStreamWaitEvent(m_stream, m_eventSino, 0);
	tBytes = sizeof(float) * iPadX * s_pTomoStack->m_aiStkSize[2];
	cudaMemcpyAsync(m_gfPadSinogram, m_pfPadSinogram, tBytes,
		cudaMemcpyDefault, m_stream);
	cudaEventRecord(m_eventSino, m_stream);	
}

void CDoSartRecon::mGetReconResult(int iLastY)
{
	if(iLastY < 0) return;
	float* pfVolXZ = s_pVolStack->GetFrame(iLastY);
	cudaStreamSynchronize(m_stream);
	//------------------------------------------------
	//  Flip z axis to match IMOD convention
	//------------------------------------------------
	int iBytes = s_pVolStack->m_aiStkSize[0] * sizeof(float);
	int iLastZ = s_pVolStack->m_aiStkSize[1] - 1;
	for(int z=0; z<=iLastZ; z++)
	{	float* pfSrc = m_pfVolXZ + z * s_pVolStack->m_aiStkSize[0];
		float* pfDst = pfVolXZ + (iLastZ - z)
			* s_pVolStack->m_aiStkSize[0];
		memcpy(pfDst, pfSrc, iBytes);
	}	
}

void CDoSartRecon::mReconstruct(int iY)
{
	size_t tBytes = s_pVolStack->GetPixels() * sizeof(float);
	cudaMemsetAsync(m_gfVolXZ, 0, tBytes, m_stream);
	m_aTomoSart.DoIt(m_gfPadSinogram, m_gfVolXZ, m_stream);
	//--------------------------------------------------
	cudaMemcpyAsync(m_pfVolXZ, m_gfVolXZ, tBytes,
		cudaMemcpyDefault, m_stream);
}

