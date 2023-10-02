#include "CMrcUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <CuUtil/DeviceArray2D.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace MrcUtil;

static __global__ void mGExtract
(	float* gfInProj,
 	float* gfCent,
	int iOutY,
	float* gfOutFrm
)
{	int i =  blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= iOutY) return;
	//--------------------
	float fX = blockIdx.x - gridDim.x * 0.5f + gfCent[0];
	float fY = i - iOutY * 0.5f + gfCent[1];
	i = i * gridDim.x + blockIdx.x;
	//-----------------------------
	gfOutFrm[i] = tex2D(texProj, fX+0.5f, fY+0.5f);
}

GExtractPatch::GExtractPatch(void)
{
	m_pTomoPatch = 0L;
}

GExtractPatch::~GExtractPatch(void)
{
	if(m_pTomoPatch != 0L) delete m_pTomoPatch;
}

void GExtractPatch::SetStack
(	CTomoStack* pTomoStack,
	CAlignParam* pAlignParam
)
{	m_pTomoStack = pTomoStack;
	m_pAlignParam = pAlignParam;
}

CTomoStack* GExtractPatch::DoIt
(	int* piStart, 
	int* piSize 
)
{	m_aiStart[0] = piStart[0];
	m_aiStart[1] = piStart[1];
	m_aiStart[2] = piStart[2];
	m_aiSize[0] = piSize[0];
	m_aiSize[1] = piSize[1];
	m_aiSize[2] = piSize[2];
	//----------------------
	m_pTomoPatch = new CTomoStack;
	m_pTomoPatch->Create(m_aiSize, true);
	mExtract();
	//---------
	CTomoStack* pTomoPatch = m_pTomoPatch;
	m_pTomoPatch = 0L;
	return pTomoPatch;
}

void GExtractPatch::mExtract(void)
{
	float afCent0[2], afCent[2];
	afCent0[0] = m_aiStart[0] + m_aiSize[0] * 0.5f;
	afCent0[1] = m_aiStart[1] + m_aiSize[1] * 0.5f;
	//---------------------------------------------
	for(int i=0; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	float fTilt = m_pAlignParam->GetTilt(i);
		float fTiltAxis = m_pAlignParam->GetTiltAxis(i);
		float* pfPatch = m_pTomoPatch->GetFrame(i);
		mCalcCenter(afCent0, fTilt, fTiltAxis, afCent);
		mExtractProj(i, afCent, pfPatch);
		m_pTomoPatch->SetCenter(i, afCent);
	}
}

void GExtractPatch::mCalcCenter
(	float* pfCent0, 
	float fTilt,
	float fTiltAxis,
	float* pfCent
)
{	double dRad = 3.14159 / 180;
	double dSin2T = sin(2 * fTiltAxis * dRad);
	double dCos2T = cos(2 * fTiltAxis * dRad);
	double dCos = cos(fTilt * dRad);
	double dP = 0.5 * (dCos + 1);
	double dM = 0.5 * (dCos - 1);
	//---------------------------
	float afMatrix[3];
	afMatrix[0] = (float)(dP + dM * dCos2T);
	afMatrix[1] = (float)(dM * dSin2T);
	afMatrix[2] = (float)(dP - dM * dCos2T);
	//--------------------------------------
	float fOrigX = m_pTomoStack->m_aiStkSize[0] * 0.5f;
	float fOrigY = m_pTomoStack->m_aiStkSize[1] * 0.5f;
	pfCent[0] = afMatrix[0] * (pfCent0[0] - fOrigX)
		+ afMatrix[1] * (pfCent0[1] - fOrigY)
		+ fOrigX;
	pfCent[1] = afMatrix[1] * (pfCent0[0] - fOrigX)
		+ afMatrix[2] * (pfCent0[1] - fOrigY)
		+ fOrigY;

	pfCent[0] = pfCent0[0];
	pfCent[1] = pfCent0[1];
} 

void GExtractPatch::mExtractProj(int iProj, float* pfCent, float* pfPatch)
{
	float* pfProj = m_pTomoStack->GetFrame(iProj);
	CDeviceArray2D aDeviceArray;
	aDeviceArray.SetFloat();
	aDeviceArray.SetChannels(1);
	aDeviceArray.Create
	( m_pTomoStack->m_aiStkSize[0],
	  m_pTomoStack->m_aiStkSize[1]
	);
	aDeviceArray.ToDevice(pfProj);
	//----------------------------
	texProj.normalized = false;
	texProj.addressMode[0] = cudaAddressModeClamp;
	texProj.addressMode[1] = cudaAddressModeClamp;
	texProj.filterMode = cudaFilterModeLinear;
	cudaBindTextureToArray(texProj, aDeviceArray.m_pCudaArray);
	//----------------------------------------------------------
	float* gfCent = 0L;
	int iBytes = sizeof(float) * 2;
	cudaMalloc(&gfCent, iBytes);
	cudaMemcpy(gfCent, pfCent, iBytes, cudaMemcpyHostToDevice);
	//---------------------------------------------------------
	int iPixels = m_aiSize[0] * m_aiSize[1];
	iBytes = iPixels * sizeof(float);
	float* gfPatch = 0L;
	cudaMalloc(&gfPatch, iBytes);
	//---------------------------
	dim3 aBlockDim(1, 512), aGridDim(m_aiSize[0], 1);
	aGridDim.y = m_aiSize[1] / aBlockDim.y + 1;
	mGExtract<<<aGridDim, aBlockDim>>>
	( gfCent, m_aiSize[1], gfPatch
	);
	cudaUnbindTexture(texProj);
	//-------------------------
	cudaMemcpy(pfPatch, gfPatch, iBytes, cudaMemcpyDeviceToHost);
	if(gfPatch != 0L) cudaFree(gfPatch);
	if(gfCent != 0L) cudaFree(gfCent);
}

