#include "CCorrectInc.h"
#include "../CFFTBuffer.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace Correct;

CFourierCropImage::CFourierCropImage(void)
{
	m_gForward2D = 0L;
	m_gInverse2D = new Util::GFFT2D;
}

CFourierCropImage::~CFourierCropImage(void)
{
	delete m_gInverse2D;
}

void CFourierCropImage::Setup(int iNthGpu, int* piImgSize, float fBin)
{
	memcpy(m_aiImgSizeIn, piImgSize, sizeof(int) * 2);
	Util::GFourierCrop2D::GetImgSize(piImgSize, fBin, m_aiImgSizeOut);
	//----------------------------------------------------------------
	CFFTBuffer* pFFTBuffer = CFFTBuffer::GetInstance();
	m_gForward2D = pFFTBuffer->GetForward2D(iNthGpu);
	m_gForward2D->CreatePlan(piImgSize, true);
	m_gInverse2D->CreatePlan(m_aiImgSizeOut, false);
}

void CFourierCropImage::DoPad(float* gfPadImgIn, float* gfPadImgOut)
{
	bool bNorm = true;
	m_gForward2D->Forward(gfPadImgIn, bNorm);
	//---------------------------------------
	bool bNormalized = bNorm;
	int aiCmpSizeIn[] = {m_aiImgSizeIn[0] / 2 + 1, m_aiImgSizeIn[1]};
	int aiCmpSizeOut[] = {m_aiImgSizeOut[0] / 2 + 1, m_aiImgSizeOut[1]};
	//------------------------------------------------------------------
	Util::GFourierCrop2D aGFourierCrop2D;
	aGFourierCrop2D.DoIt((cufftComplex*)gfPadImgIn, aiCmpSizeIn, 
	  bNormalized, (cufftComplex*)gfPadImgOut, aiCmpSizeOut);
	//-------------------------------------------------------
	m_gInverse2D->Inverse((cufftComplex*)gfPadImgOut);
}

