#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>

using namespace Util;

CPad2D::CPad2D(void)
{
}

CPad2D::~CPad2D(void)
{
}

void CPad2D::GetPadSize(int* piImgSize, int* piPadSize)
{
	piPadSize[0] = (piImgSize[0] / 2 + 1) * 2;
	piPadSize[1] = piImgSize[1];
}

void CPad2D::GetImgSize(int* piPadSize, int* piImgSize)
{
	piImgSize[0] = (piPadSize[0] / 2 - 1) * 2;
	piImgSize[1] = piPadSize[1];
}

float* CPad2D::GGetPadBuf(int* piImgSize, bool bZero)
{
	int aiPadSize[2] = {0};
	CPad2D::GetPadSize(piImgSize, aiPadSize);
	//---------------------------------------
	float* gfPadBuf = mGGetBuf(aiPadSize, bZero);
	return gfPadBuf;
}

float* CPad2D::CGetPadBuf(int* piImgSize, bool bZero)
{
	int aiPadSize[2] = {0};
	CPad2D::GetPadSize(piImgSize, aiPadSize);
	//---------------------------------------
	float* pfPadBuf = mCGetBuf(aiPadSize, bZero);
	return pfPadBuf;
}

float* CPad2D::GGetImgBuf(int* piPadSize, bool bZero)
{
	int aiImgSize[2] = {0};
	CPad2D::GetImgSize(piPadSize, aiImgSize);
	//---------------------------------------
	float* gfImgBuf = mGGetBuf(aiImgSize, bZero);
	return gfImgBuf;
}

float* CPad2D::CGetImgBuf(int* piPadSize, bool bZero)
{
	int aiImgSize[2] = {0};
	CPad2D::GetImgSize(piPadSize, aiImgSize);
	//---------------------------------------
	float* pfImgBuf = mCGetBuf(aiImgSize, bZero);
	return pfImgBuf;
}

float* CPad2D::GPad(float* pfImg, int* piSize)
{
	float* gfPad = CPad2D::GGetPadBuf(piSize, false);
	this->Pad(pfImg, piSize, gfPad);
	return gfPad;
}

float* CPad2D::CPad(float* pfImg, int* piSize)
{
	float* pfPad = CPad2D::CGetPadBuf(piSize, false);
	this->Pad(pfImg, piSize, pfPad);
	return pfPad; 
}

void CPad2D::Pad(float* pfImg, int* piImgSize, float* pfPad)
{
	int iBytes = piImgSize[0] * sizeof(float);
	int iPadX = (piImgSize[0] / 2 + 1) * 2;
	//-------------------------------------
	for(int y=0; y<piImgSize[1]; y++)
	{	float* pfSrc = pfImg + y * piImgSize[0];
		float* pfDst = pfPad + y * iPadX;
		cudaMemcpy(pfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

float* CPad2D::GUnpad(float* pfPad, int* piSize)
{
	float* gfImg = CPad2D::GGetImgBuf(piSize, false);
	this->Unpad(pfPad, piSize, gfImg);
	return gfImg;
}

float* CPad2D::CUnpad(float* pfPad, int* piSize)
{
	float* pfImg = CPad2D::CGetImgBuf(piSize, false);
	this->Unpad(pfPad, piSize, pfImg);
	return pfImg;
}

void CPad2D::Unpad(float* pfPad, int* piPadSize, float* pfImg)
{
	int iImageX = (piPadSize[0] / 2 - 1) * 2;
	int iBytes = iImageX * sizeof(float);
	//-----------------------------------
	for(int y=0; y<piPadSize[1]; y++)
	{	float* pfSrc = pfPad + y * piPadSize[0];
		float* pfDst = pfImg + y * iImageX;
		cudaMemcpy(pfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

float* CPad2D::mGGetBuf(int* piSize, bool bZero)
{
	size_t tBytes = sizeof(float) * piSize[0] * piSize[1];
	float* gfBuf = 0L;
	cudaMalloc(&gfBuf, tBytes);
	if(!bZero) return gfBuf;
	//----------------------
	cudaMemset(gfBuf, 0, tBytes);
	return gfBuf;
}

float* CPad2D::mCGetBuf(int* piSize, bool bZero)
{
	int iPixels = piSize[0] * piSize[1];
	float* pfBuf = new float[iPixels];
	if(!bZero) return pfBuf;
	//----------------------
	memset(pfBuf, 0, sizeof(float) * iPixels);
	return pfBuf;
}
