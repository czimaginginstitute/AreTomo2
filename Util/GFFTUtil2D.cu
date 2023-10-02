#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace Util;

static __global__ void mGMultiplyFactor
(	cufftComplex* gCmp,
	int iCmpY,
	float fFactor
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	gCmp[i].x *= fFactor;
	gCmp[i].y *= fFactor;
}

static __global__ void mGGetAmp
(	cufftComplex* gCmp,
	int iCmpY,
	float* gfAmp
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	gfAmp[i] = sqrtf(gCmp[i].x * gCmp[i].x + gCmp[i].y * gCmp[i].y);
}

static __global__ void mGShiftFrame
(       cufftComplex* gComp,
        float fShiftX,
        float fShiftY,
        int iCmpY
)
{       int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	if(y  > (iCmpY / 2)) y -= iCmpY;
        float fPhaseShift = blockIdx.x * fShiftX + y * fShiftY;
        float fCos = cosf(fPhaseShift);
        float fSin = sinf(fPhaseShift);
        //-----------------------------
	float fRe = fCos * gComp[i].x - fSin * gComp[i].y;
        float fIm = fCos * gComp[i].y + fSin * gComp[i].x;
        gComp[i].x = fRe;
        gComp[i].y = fIm;
}

GFFTUtil2D::GFFTUtil2D(void)
{
}

GFFTUtil2D::~GFFTUtil2D(void)
{
}

void GFFTUtil2D::Multiply
( 	cufftComplex* gComp,
	int* piCmpSize,
	float fFactor
)
{	dim3 aBlockDim(1, 512);
	int iGridY = piCmpSize[1] / aBlockDim.y + 1;
	dim3 aGridDim(piCmpSize[0], iGridY);
	//----------------------------------
	mGMultiplyFactor<<<aGridDim, aBlockDim>>>
	(  gComp, 
	   piCmpSize[1],
	   fFactor
	);
}

void GFFTUtil2D::GetAmp
(	cufftComplex* gComp,
	int* piCmpSize,
	float* pfAmpRes,
	bool bGpuRes
)
{	size_t tCmpSize = piCmpSize[0] * (size_t)piCmpSize[1];
	size_t tBytes = sizeof(float) * tCmpSize;
	float* gfAmp = 0L;
	if(bGpuRes) gfAmp = pfAmpRes;
	else cudaMalloc(&gfAmp, tBytes);
	//------------------------------
	dim3 aBlockDim(1, 512);
	int iGridY = piCmpSize[1] / aBlockDim.y + 1;
	dim3 aGridDim(piCmpSize[0], iGridY);
	//----------------------------------
	mGGetAmp<<<aGridDim, aBlockDim>>>
	(  gComp, piCmpSize[1], gfAmp
	);
	//---------------------------
	if(bGpuRes) return;
	cudaMemcpy(pfAmpRes, gfAmp, tBytes, cudaMemcpyDeviceToHost);
	cudaFree(gfAmp);
}
	
void GFFTUtil2D::Shift
(       cufftComplex* gComp,
	int* piCmpSize,
        float* pfShift
)
{       if(pfShift == 0L) return;
        if(pfShift[0] == 0.0f && pfShift[1] == 0.0f) return;
        //--------------------------------------------------
	dim3 aBlockDim(1, 512);
        int iGridY = piCmpSize[1] / aBlockDim.y + 1;
        dim3 aGridDim(piCmpSize[0], iGridY);
        //----------------------------------
	int iNx = 2 * (piCmpSize[0] - 1);
        float f2PI = (float)(8 * atan(1.0));
        float fShiftX = pfShift[0] * f2PI / iNx;
        float fShiftY = pfShift[1] * f2PI / piCmpSize[1];
        //-----------------------------------------------
	mGShiftFrame<<<aGridDim, aBlockDim>>>
        (  gComp, fShiftX, fShiftY, piCmpSize[1]
        );
}
