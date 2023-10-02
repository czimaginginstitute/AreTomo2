#include "CUtilInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace Util;

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
static __global__ void mGAddImages
(	float* gfImage1,
	float fFactor1,
	float* gfImage2,
	float fFactor2,
	float* gfSum,
	int iSizeY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * gridDim.x + blockIdx.x;
	gfSum[i] = gfImage1[i] * fFactor1 + gfImage2[i] * fFactor2;
}

GAddImages::GAddImages(void)
{
}

GAddImages::~GAddImages(void)
{
}

void GAddImages::DoIt
(	float* gfImage1,
	float fFactor1,
	float* gfImage2,
	float fFactor2,
	float* gfSum,
	int* piImgSize	 
)
{	dim3 aBlockDim(1, 512);
	int iGridY = piImgSize[1] / aBlockDim.y + 1;
        dim3 aGridDim(piImgSize[0], iGridY);
        mGAddImages<<<aGridDim, aBlockDim>>>
	(  gfImage1, fFactor1,
	   gfImage2, fFactor2,
	   gfSum, piImgSize[1]
	);
}

