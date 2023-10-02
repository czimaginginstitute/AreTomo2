#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace Util;

static __global__ void mGCrop
( 	cufftComplex* gCmpIn, 
	int iSizeInX,
	int iSizeInY,
	float fScale,
  	cufftComplex* gCmpOut, 
	int iSizeOutY
)
{	int yOut = blockIdx.y * blockDim.y + threadIdx.y;
	if(yOut >= iSizeOutY) return;
	//---------------------------
	int yIn = yOut;
        if(yOut > (iSizeOutY / 2)) yIn = yOut - iSizeOutY + iSizeInY;
	//-----------------------------------------------------------
	int iIn = yIn * iSizeInX + blockIdx.x;
	int iOut = yOut * gridDim.x + blockIdx.x;
	gCmpOut[iOut].x = gCmpIn[iIn].x * fScale;
	gCmpOut[iOut].y = gCmpIn[iIn].y * fScale;
}

GFourierCrop2D::GFourierCrop2D(void)
{
}

GFourierCrop2D::~GFourierCrop2D(void)
{
}

void GFourierCrop2D::GetImgSize
(	int* piImgSizeIn,
	float fBin,
	int* piImgSizeOut
)
{	piImgSizeOut[0] = piImgSizeIn[0];
	piImgSizeOut[1] = piImgSizeIn[1];
	if(fBin == 1) return;
	//-------------------
	piImgSizeOut[0] = ((int)(piImgSizeIn[0] / fBin + 0.1f)) / 2 * 2;
	piImgSizeOut[1] = ((int)(piImgSizeIn[1] / fBin + 0.1f)) / 2 * 2;
}

void GFourierCrop2D::GetPadSize
(	int* piPadSizeIn,
	float fBin,
	int* piPadSizeOut
)
{	int aiImgSizeIn[] = {1, piPadSizeIn[1]};
	aiImgSizeIn[0] = (piPadSizeIn[0] / 2 - 1) * 2;
	GFourierCrop2D::GetImgSize(aiImgSizeIn, fBin, piPadSizeOut);
	piPadSizeOut[0] = (piPadSizeOut[0] / 2 + 1) * 2;
}

void GFourierCrop2D::GetCmpSize
(	int* piCmpSizeIn,
	float fBin,
	int* piCmpSizeOut
)
{	piCmpSizeOut[0] = piCmpSizeIn[0];
	piCmpSizeOut[1] = piCmpSizeIn[1];
	if(fBin == 1) return;
	//-------------------
	int aiImgSizeIn[] = {(piCmpSizeIn[0] - 1) * 2, piCmpSizeIn[1]};
	GFourierCrop2D::GetImgSize(aiImgSizeIn, fBin, piCmpSizeOut);
	piCmpSizeOut[0] = piCmpSizeOut[0] / 2 + 1;
}

void GFourierCrop2D::CalcBinning
(	int* piSizeIn,
	int* piSizeOut,
	bool bImgSize,
	float* pfBinning
)
{	if(bImgSize) pfBinning[0] = piSizeIn[0] / (float)piSizeOut[0];
	else pfBinning[0] = (piSizeIn[0] - 1.0f) / (piSizeOut[0] - 1);
	pfBinning[1] = piSizeIn[1] / (float)piSizeOut[1];
}

void GFourierCrop2D::DoIt
( 	cufftComplex* gCmpIn, 
	int* piSizeIn,
	bool bNormalized,
  	cufftComplex* gCmpOut, 
	int* piSizeOut
)
{	float fScale = 1.0f;
	if(!bNormalized)
	{	float fSizeIn = piSizeIn[0] * piSizeIn[1];
		float fSizeOut = piSizeOut[0] * piSizeOut[1];
		fScale = fSizeOut / fSizeIn;
	}
	//----------------------------------
	dim3 aBlockDim(1, 1024);
	dim3 aGridDim(piSizeOut[0], piSizeOut[1] / aBlockDim.y + 1);
	//----------------------------------------------------------
	mGCrop<<<aGridDim, aBlockDim>>>
	( gCmpIn, piSizeIn[0], piSizeIn[1], fScale,
	  gCmpOut, piSizeOut[1]
	);
}

