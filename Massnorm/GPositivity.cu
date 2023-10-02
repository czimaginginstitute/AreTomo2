#include "CMassNormInc.h"
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace MassNorm;

static __global__ void mGSubtract(float* gfImg, int iPixels, float fVal)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iPixels) return;
	else if(gfImg[i] < (float)-1e10) return;
	else gfImg[i] -= fVal;
}

GPositivity::GPositivity(void)
{
}

GPositivity::~GPositivity(void)
{
}

void GPositivity::DoIt(MrcUtil::CTomoStack* pTomoStack)
{
	printf("Set positivity ...\n");
	int* piStkSize = pTomoStack->m_aiStkSize;
	//---------------------------------------
	float *gfFrm = 0L;
	int iPixels = piStkSize[0] * piStkSize[1];
	size_t tBytes = sizeof(float) * iPixels;
	cudaMalloc(&gfFrm, tBytes);
	//-------------------------
	bool bPadded = true, bSync = true;
	float fStackMin = (float)1e30;
	Util::GFindMinMax2D aFindMinMax;
	aFindMinMax.SetSize(piStkSize, !bPadded);
	//---------------------------------------
	for(int i=0; i<piStkSize[2]; i++)
	{	float* pfFrm = pTomoStack->GetFrame(i);
		cudaMemcpy(gfFrm, pfFrm, tBytes, cudaMemcpyDefault);
		//--------------------------------------------------
		float fMin = aFindMinMax.DoMin(gfFrm, bSync);
		float fMax = aFindMinMax.DoMax(gfFrm, bSync);
		if(fStackMin  > fMin) fStackMin = fMin;
		printf("%4d  %8.2f  %8.2f\n", i, fMin, fMax);
	}
	if(fStackMin >= 0)
	{	cudaFree(gfFrm);
		printf("Positivity set.\n\n");
		return;
	}
	//-------------
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = (iPixels + aBlockDim.x - 1) / aBlockDim.x;
	for(int i=0; i<piStkSize[2]; i++)
	{	float* pfFrm = pTomoStack->GetFrame(i);
		cudaMemcpy(gfFrm, pfFrm, tBytes, cudaMemcpyDefault);
		mGSubtract<<<aGridDim, aBlockDim>>>(gfFrm, iPixels, fStackMin);
		cudaMemcpy(pfFrm, gfFrm, tBytes, cudaMemcpyDefault);
	}
	cudaFree(gfFrm);
	printf("Positivity set.\n\n");
}

