#include "CCTFCorInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace CTFCor;


static __global__ void mGCorBilinear
(	int iCmpY,
	cufftComplex* gCmpImg
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	if(blockIdx.x == 0 && y == 0) return;
	//-----------------------------------
	float fX = blockIdx.x * 0.5f / (gridDim.x - 1);
	float fY = y / (float)iCmpY;
	if(fY > 0.5f) fY -= 1.0f;
	//-----------------------
	fX = sqrtf(fX * fX + fY * fY) * 3.141592654f;
	fX = sinf(fX) / fX;  // sinc(PI * f)
	fX = fX * fX; 
	//-----------
	int i = y * gridDim.x + blockIdx.x;
	gCmpImg[i].x /= fX;
	gCmpImg[i].y /= fX;
}

GCorBilinear::GCorBilinear(void)
{
}

GCorBilinear::~GCorBilinear(void)
{
}

void GCorBilinear::DoIt
(	cufftComplex* gCmpImg,
	int* piCmpSize
)
{	dim3 aBlockDim(1, 1024);
	dim3 aGridDim(piCmpSize[0], piCmpSize[1] / aBlockDim.y + 1);
	//----------------------------------------------------------	
	mGCorBilinear<<<aGridDim, aBlockDim>>>
	( piCmpSize[1], gCmpImg
	);
}

