#include "CPatchAlignInc.h"
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace PatchAlign;

static __global__ void mGAddFloat
(	float* gfImg1, float fFactor1,
	float* gfImg2, float fFactor2,
	float* gfSum, int nxy
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nxy) return;
	gfSum[i] = gfImg1[i] * fFactor1 + gfImg2[i] * fFactor2;
}

GAddImages::GAddImages(void)
{
}

GAddImages::~GAddImages(void)
{
}

void GAddImages::DoIt
(	float* gfImg1,
	float fFactor1,
	float* gfImg2,
	float fFactor2,
	float* gfSum,
	int* piFrmSize,
        cudaStream_t stream
)
{       int nxy = piFrmSize[0] * piFrmSize[1];
        dim3 aBlockDim(512, 1, 1);
        dim3 aGridDim((nxy + aBlockDim.x - 1) / aBlockDim.x, 1, 1);
	mGAddFloat<<<aGridDim, aBlockDim, 0, stream>>>
	( gfImg1, fFactor1,
	  gfImg2, fFactor2,
	  gfSum, nxy
	);
}

