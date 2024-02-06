#include "CStreAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace StreAlign;

CStretchXcf::CStretchXcf(void)
{
	m_gCmpRef = 0L;
	m_gCmp = 0L;
	m_gCmpBuf = 0L;
	m_fBFactor = 200.0f;
}

CStretchXcf::~CStretchXcf(void)
{
	this->Clean();
}

void CStretchXcf::Clean(void)
{
	m_fft2D.DestroyPlan();
	m_xcf2D.Clean();
	//--------------
	if(m_gCmpRef != 0L) cudaFree(m_gCmpRef);
	m_gCmpRef = 0L;
}

void CStretchXcf::Setup
(	int* piImgSize,
	float fBFactor
)
{	this->Clean();
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_fBFactor = fBFactor;
	//-----------------
	m_aiPadSize[0] = (m_aiImgSize[0] / 2 + 1) * 2;
	m_aiPadSize[1] = m_aiImgSize[1];
	m_aiCmpSize[0] = m_aiPadSize[0] / 2;
	m_aiCmpSize[1] = m_aiPadSize[1];
	//-----------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	size_t tBytes = sizeof(cufftComplex) * iCmpSize * 3;
	cudaMalloc(&m_gCmpRef, tBytes);
	//-----------------
	m_gCmp = m_gCmpRef + iCmpSize;
	m_gCmpBuf = m_gCmp + iCmpSize;
	//-----------------
	bool bPadded = true;
	m_xcf2D.Setup(m_aiCmpSize);
	//-----------------
	bool bForward = true;
	m_fft2D.CreatePlan(m_aiImgSize, bForward);
}

void CStretchXcf::DoIt
(	float* pfRefImg,
	float* pfImg,
	float fRefTilt,
	float fTilt,
	float fTiltAxis
)
{	mPadImage(pfRefImg, (float*)m_gCmpRef);
	mNormalize((float*)m_gCmpRef);
	//-----------------
	mPadImage(pfImg, (float*)m_gCmpBuf);
	bool bPadded = true;
	bool bRandomFill = true;
	double dRad = 4.0 * atan(1.0) / 180.0;
	double dStretch = cos(dRad * fRefTilt) / cos(dRad * fTilt);
	Util::GStretch aGStretch;
	aGStretch.DoIt
	( (float*)m_gCmpBuf, m_aiPadSize, bPadded, (float)dStretch,
	  fTiltAxis, (float*)m_gCmp, bRandomFill
	);
	mNormalize((float*)m_gCmp);
	//-----------------
	mRoundEdge();
	mForwardFFT();
	m_xcf2D.DoIt(m_gCmpRef, m_gCmp, m_fBFactor);
	//-----------------
	float fPeak = m_xcf2D.SearchPeak();
	m_xcf2D.GetShift(m_afShift, 1.0f);
	//------------------------------------------------------
	// Prevent absurd large shift from causing trouble to
	// downstream finding tilt angle offset.
	//------------------------------------------------------
	if(fabs(m_afShift[0]) > (0.25 * m_aiImgSize[0]) ||
	   fabs(m_afShift[1]) > (0.25 * m_aiImgSize[1]))
	{	m_afShift[0] = 0.0f;
		m_afShift[1] = 0.0f;
	}
}

void CStretchXcf::GetShift(float fFactX, float fFactY, float* pfShift)
{
	pfShift[0] = fFactX * m_afShift[0];
	pfShift[1] = fFactY * m_afShift[1];
}

void CStretchXcf::mPadImage(float* pfImg, float* gfPadImg)
{
	size_t tBytes = sizeof(float) * m_aiImgSize[0];
	for(int y=0; y<m_aiImgSize[1]; y++)
	{	float* pfSrc = pfImg + y * m_aiImgSize[0];
		float* gfDst = gfPadImg + y * m_aiPadSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	}
}

void CStretchXcf::mNormalize(float* gfPadImg)
{
	bool bPadded = true;
	float afMeanStd[] = {0.0f, 1.0f};
	Util::GCalcMoment2D aGCalcMoment2D;
	aGCalcMoment2D.SetSize(m_aiPadSize, bPadded);
	afMeanStd[0] = aGCalcMoment2D.DoIt(gfPadImg, 1, true);
	afMeanStd[1] = aGCalcMoment2D.DoIt(gfPadImg, 2, true)
	   - afMeanStd[0] * afMeanStd[0];
	if(afMeanStd[1] <= 0) afMeanStd[1] = 0.0f;
	else afMeanStd[1] = (float)sqrtf(afMeanStd[1]);
	//---------------------------------------------
	Util::GNormalize2D aGNorm2D;
	aGNorm2D.DoIt(gfPadImg, m_aiPadSize, bPadded,
		afMeanStd[0], afMeanStd[1]);
}

void CStretchXcf::mRoundEdge(void)
{
	float afCent[] = {m_aiImgSize[0] * 0.5f, m_aiImgSize[1] * 0.5f};
	float afSize[] = {m_aiImgSize[0] * 1.0f, m_aiImgSize[1] * 1.0f};
	//--------------------------------------------------------------
	bool bPadded = true;
	Util::GRoundEdge roundEdge;
	float fPower = 4.0f;
	roundEdge.SetMask(afCent, afSize);
	roundEdge.DoIt((float*)m_gCmpRef, m_aiPadSize, bPadded, fPower);
	roundEdge.DoIt((float*)m_gCmp, m_aiPadSize, bPadded, fPower);
}

void CStretchXcf::mForwardFFT(void)
{
	bool bNorm = true;
	m_fft2D.Forward((float*)m_gCmpRef, bNorm);
	m_fft2D.Forward((float*)m_gCmp, bNorm);
}
