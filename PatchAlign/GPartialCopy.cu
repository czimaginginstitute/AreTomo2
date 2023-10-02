#include "CPatchAlignInc.h"
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace PatchAlign;

template <typename T>
static __global__ void mGPartialCopy
(	T* gSrc, int iSrcX, 
	T* gDst, int iDstX,
	int iCpyX
)
{	int iX = blockIdx.x * blockDim.x + threadIdx.x;
	if(iX >= iCpyX) return;
  	gDst[blockIdx.y * iDstX + iX] = gSrc[blockIdx.y * iSrcX + iX];
}

GPartialCopy::GPartialCopy(void)
{
}

GPartialCopy::~GPartialCopy(void)
{
}

void GPartialCopy::DoIt
(	float* gfSrc,
	int iSrcSizeX,
	float* gfDst,
	int* piDstSize,
	int iCpySizeX,
	cudaStream_t stream
)
{	dim3 aBlockDim(64, 1, 1);
        dim3 aGridDim(1, piDstSize[1], 1);
	aGridDim.x = (iCpySizeX + aBlockDim.x - 1) / aBlockDim.x;;
        mGPartialCopy<<<aGridDim, aBlockDim, 0, stream>>>(gfSrc,
		iSrcSizeX, gfDst, piDstSize[0], iCpySizeX);
}

void GPartialCopy::DoIt
(	float* gfSrcImg,
	int* piSrcSize,
	int* piSrcStart,
	float* gfPatch,
	int* piPatSize,
	bool bPadded,
	cudaStream_t stream
)
{	int iOffset = piSrcStart[1] * piSrcSize[0] + piSrcStart[0];
	float* gfSrc = gfSrcImg + iOffset;
	//---------------------------
	int iCpySizeX = piPatSize[0];
	if(bPadded) iCpySizeX = (piPatSize[0] / 2 - 1) * 2;
	this->DoIt(gfSrc, piSrcSize[0], gfPatch, piPatSize, 
		iCpySizeX, stream);
}
