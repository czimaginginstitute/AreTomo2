#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <math.h>

using namespace FindCtf;

static __global__ void mGDoBFactor
(	cufftComplex* gInCmp, 
	int iCmpY,
	float fScale,
	cufftComplex* gOutCmp
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	if(y > (iCmpY / 2)) y -= iCmpY;
	float fFilt = expf(fScale * (blockIdx.x * blockIdx.x + y * y));
	gOutCmp[i].x = gInCmp[i].x * fFilt;
	gOutCmp[i].y = gInCmp[i].y * fFilt;
}

static __global__ void mGDoCutoff
(	cufftComplex* gInCmp,
	int iCmpY,
	float fCutoff,
	cufftComplex* gOutCmp
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
        //---------------------------------
	if(y > (iCmpY / 2)) y -= iCmpY;
	float fX = blockIdx.x * 0.5f / (gridDim.x - 1);
	float fY = y / (float)iCmpY;
	float fR = sqrtf(fX * fX + fY * fY) / fCutoff;
	//--------------------------------------------
	float fFilt = 0.0f;
	if(fR < 1)
	{	fR = 0.5f * (1.0f - cosf(3.1416f * fR));
		fFilt = 1.0f - powf(fR, 20.0f);
	}
	gOutCmp[i].x = gInCmp[i].x * fFilt;
	gOutCmp[i].y = gInCmp[i].y * fFilt;
}

GLowpass2D::GLowpass2D(void)
{
}

GLowpass2D::~GLowpass2D(void)
{
}

void GLowpass2D::DoBFactor
(	cufftComplex* gInCmp,
	cufftComplex* gOutCmp,
	int* piCmpSize,
	float fBFactor
)
{	int iNx = (piCmpSize[0] - 1) * 2;
	double dTemp = iNx * iNx + piCmpSize[1] * piCmpSize[1];
	float fScale = (float)(-fBFactor / dTemp);
	//----------------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piCmpSize[0], 1);
	aGridDim.y = piCmpSize[1] / aBlockDim.y + 1;
	mGDoBFactor<<<aGridDim, aBlockDim>>>
	(  gInCmp, piCmpSize[1], fScale, 
	   gOutCmp
	);
}

cufftComplex* GLowpass2D::DoBFactor
(	cufftComplex* gCmp,
	int* piCmpSize,
	float fBFactor
)
{	size_t tBytes = sizeof(cufftComplex);
	tBytes *= (piCmpSize[0] * piCmpSize[1]);
	cufftComplex* gOutCmp = 0L;
	cudaMalloc(&gOutCmp, tBytes);
	this->DoBFactor(gCmp, gOutCmp, piCmpSize, fBFactor);
	return gOutCmp;
}

void GLowpass2D::DoCutoff
(	cufftComplex* gInCmp,
	cufftComplex* gOutCmp,
	int* piCmpSize,
	float fCutoff
)
{	dim3 aBlockDim(1, 512);
        dim3 aGridDim(piCmpSize[0], 1);
        aGridDim.y = piCmpSize[1] / aBlockDim.y + 1;
	mGDoCutoff<<<aGridDim, aBlockDim>>>
	(  gInCmp, piCmpSize[1], fCutoff,
	   gOutCmp
	);
}

cufftComplex* GLowpass2D::DoCutoff
(	cufftComplex* gCmp,
	int* piCmpSize,
	float fCutoff
)
{	size_t tBytes = sizeof(cufftComplex);
        tBytes *= (piCmpSize[0] * piCmpSize[1]);
        cufftComplex* gOutCmp = 0L;
        cudaMalloc(&gOutCmp, tBytes);
	this->DoCutoff(gCmp, gOutCmp, piCmpSize, fCutoff);
	return gOutCmp;
}

