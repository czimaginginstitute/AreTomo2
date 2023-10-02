#include "CCorrectInc.h"
#include "../CFFTBuffer.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace Correct;

CCorrLinearInterp::CCorrLinearInterp(void)
{
	m_gForward2D = new CuUtilFFT::GFFT2D;
	m_gInverse2D = new CuUtilFFT::GFFT2D;
}

CCorrLinearInterp::~CCorrLinearInterp(void)
{
	delete m_gForward2D;
	delete m_gInverse2D;
}

void CCorrLinearInterp::Setup(int* piImgSize)
{
	memcpy(m_aiImgSize, piImgSize, sizeof(int) * 2);
	memcpy(m_aiCmpSize, piImgSize, sizeof(int) * 2);
	m_aiCmpSize[0] = m_aiImgSize[0] / 2 + 1;
	//----------------------------------------------
	m_gForward2D->CreatePlan(m_aiImgSize, true);
	m_gInverse2D->CreatePlan(m_aiImgSize, false);
}
/*
void CCorrLinearInterp::DoIt(float* gfImg, float* gfPadBuf)
{
	bool bNorm = true;
	cufftComplex* gCmpImg = (cufftComplex*)gfPadBuf;
	m_gForward2D->Forward(gfImg, gCmpImg, bNorm);
	//-------------------------------------------
	Util::GCorrLinearInterp gCorrLinearInterp;
        gCorrLinearInterp.DoIt(gCmpImg, m_aiCmpSize, 0);
	//----------------------------------------------
	m_gInverse2D->Inverse(gCmpImg, gfImg);
}
*/

void CCorrLinearInterp::DoIt(float* gfImg, float* gfPadBuf)
{
	
}
