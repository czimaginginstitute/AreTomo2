#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace CommonLine;

static __global__ void mGAdd
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	cufftComplex* gSum,
	int iSign,
	int iCmpSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iCmpSize) return;
	//-----------------------
	gSum[i].x = gCmp1[i].x + iSign * gCmp2[i].x;
	gSum[i].y = gCmp1[i].y + iSign * gCmp2[i].y;
}

GAddTwoLines::GAddTwoLines(void)
{
}

GAddTwoLines::~GAddTwoLines(void)
{
}

void GAddTwoLines::DoIt
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	cufftComplex* gSum,
	int iOperator,		// 1: add, -1: subtract
	int iCmpSize	
)
{	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = iCmpSize / aBlockDim.x + 1;
	mGAdd<<<aGridDim, aBlockDim>>>
	(  gCmp1, gCmp2, gSum, iOperator, iCmpSize
	);
}

