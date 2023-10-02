#include "CProjAlignInc.h"
#include "../Util/CUtilInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <memory.h>
#include <stdio.h>

using namespace Projalign;

CCentralXcf::CCentralXcf(void)
{
	m_pfXcfImg = 0L;
}

CCentralXcf::~CCentralXcf(void)
{
	if(m_pfXcfImg != 0L) delete[] m_pfXcfImg;
}

void CCentralXcf::DoIt
(	float* pfRef,
	bool bRefGpu,
	float* pfImg,
	bool bImgGpu,
	int* piImgSize
)
{	float* gfRef = mGetCentral(pfRef, bRefGpu, piImgSize);	
	float* gfImg = mGetCentral(pfImg, bImgGpu, piImgSize);
	//----------------------------------------------------
	m_iXcfBin = (int)(m_aiImgSize[0] / 1024.0f + 0.5f);
	if(m_iXcfBin < 1) m_iXcfBin = 1;
	float* gfPadRef = mBinImage(gfRef);
	if(gfRef != 0L) cudaFree(gfRef);
	float* gfPadImg = mBinImage(gfImg);
	if(gfImg != 0L) cudaFree(gfImg);
	//------------------------------
	mRemoveMean(gfPadRef);
	mRemoveMean(gfPadImg);
	//--------------------
	mRoundEdge(gfPadRef);	
	mRoundEdge(gfPadImg);
	//-------------------
	mCorrelate(gfPadRef, gfPadImg);
	if(gfPadRef != 0L) cudaFree(gfPadRef);
	if(gfPadImg != 0L) cudaFree(gfPadImg);
}

void CCentralXcf::GetShift(float* pfShift)
{
	pfShift[0] = m_afShift[0];
	pfShift[1] = m_afShift[1];
}

float* CCentralXcf::mGetCentral(float* pfImg, bool bGpu, int* piImgSize)
{
	m_aiImgSize[0] = piImgSize[0] / 2 * 2;
	m_aiImgSize[1] = piImgSize[1] / 2 * 2;
	int iPixels = m_aiImgSize[0] * m_aiImgSize[1];
	//--------------------------------------------
	float* gfBuf = 0L;
	size_t tBytes = sizeof(float) * iPixels;
	cudaMalloc(&gfBuf, tBytes);
	//-------------------------
	tBytes = sizeof(float) * m_aiImgSize[0];
	int iX = (piImgSize[0] - m_aiImgSize[0]) / 2;
	int iY = (piImgSize[1] - m_aiImgSize[1]) / 2;
	int iOffset = iY * piImgSize[0] + iX;
	cudaMemcpyKind aKind = cudaMemcpyDeviceToDevice;
	if(!bGpu) aKind = cudaMemcpyHostToDevice;
	//---------------------------------------
	for(int y=0; y<m_aiImgSize[1]; y++)
	{	float* pfSrc = pfImg + y * piImgSize[0] + iOffset;
		float* gfDst = gfBuf + y * m_aiImgSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, aKind);
	}
	return gfBuf;
}

float* CCentralXcf::mBinImage(float* gfImg)
{
	int aiBinning[] = {m_iXcfBin, m_iXcfBin};
	Util::GBinImage2D aGBinImg2D;
	aGBinImg2D.Setup(m_aiImgSize, aiBinning);
	aGBinImg2D.GetBinSize(m_aiXcfSize);
	aGBinImg2D.GetPadSize(m_aiPadSize);
	//---------------------------------
	bool bZero = true, bGpu = true;
	float* gfPadImg = aGBinImg2D.GetBuf(m_aiPadSize, !bZero, bGpu);
	aGBinImg2D.DoPad(gfImg, bGpu, gfPadImg, bGpu);
	return gfPadImg;
}

void CCentralXcf::mRemoveMean(float* gfPadImg)
{
	bool bGpu = true;
	Util::GRemoveMean aGRemoveMean;
	aGRemoveMean.DoPad(gfPadImg, bGpu, m_aiPadSize);
}

void CCentralXcf::mRoundEdge(float* gfPadImg)
{
	float afCent[] = {0.0f, 0.0f};
	float afMaskSize[] = {0.0f, 0.0f}; 
	afCent[0] = m_aiXcfSize[0] * 0.5f;
	afCent[1] = m_aiXcfSize[1] * 0.5f;
	afMaskSize[0] = m_aiXcfSize[0] * 1.00f;
	afMaskSize[1] = m_aiXcfSize[1] * 1.00f;
	//-------------------------------------
	Util::GRoundEdge aGRoundEdge;
	aGRoundEdge.SetMask(afCent, afMaskSize);
	aGRoundEdge.DoIt(gfPadImg, m_aiPadSize);	
}

void CCentralXcf::mCorrelate(float* gfPadRef, float* gfPadImg)
{
	bool bNorm = true;
	CuUtilFFT::GFFT2D aGFFT2D;
	aGFFT2D.Forward(gfPadRef, m_aiPadSize, !bNorm);
	aGFFT2D.Forward(gfPadImg, m_aiPadSize, !bNorm);
	int aiCmpSize[] = {m_aiPadSize[0]/2, m_aiPadSize[1]};
	cufftComplex* gRefCmp = (cufftComplex*)gfPadRef;
	cufftComplex* gImgCmp = (cufftComplex*)gfPadImg;
	//----------------------------------------------
	Util::GXcf2D aGXcf2D;
	aGXcf2D.SetBFactor(300.0f);
	aGXcf2D.DoIt(gRefCmp, gImgCmp, aiCmpSize);
	//----------------------------------------
	bool bClean = true;
	aGXcf2D.SearchPeak();
	if(m_pfXcfImg != 0L) delete[] m_pfXcfImg;
	m_pfXcfImg = aGXcf2D.GetXcfImg(bClean);
	m_aiXcfSize[0] = aGXcf2D.m_aiXcfSize[0];
	m_aiXcfSize[1] = aGXcf2D.m_aiXcfSize[1];
	//--------------------------------------
	aGXcf2D.GetShift(m_afShift, m_iXcfBin);
}
