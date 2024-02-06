#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../StreAlign/CStreAlignInc.h"
#include <Util/Util_Thread.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace TiltOffset
{
class CStretchCC2D
{
public:
	CStretchCC2D(void);
	~CStretchCC2D(void);
	void Clean(void);
	void SetSize(int* piSize, bool bPadded);
	float DoIt
	( float* pfRefImg,// lower tilt image
	  float* pfImg,   // higher tilt image to be stretched
	  float fRefTilt,
	  float fImgTilt,
	  float* pfTiltOffsets,
	  int iNumOffsets
	);
	float* m_pfCCs;
	int m_iNumOffsets;
private:
        int m_aiImgSize[2];
	int m_iImageX;
	int m_aiCCSize[2];
        float* m_gfImg1;
        float* m_gfImg2;
	float* m_gfStreImg;
	float* m_gfCCImg1;
	float* m_gfCCImg2;
	float m_afTilt[2];
	float* m_pfTiltOffsets;
};

class CTiltOffsetMain
{
public:
	CTiltOffsetMain(void);
	~CTiltOffsetMain(void);
	void Setup(int* piStkSize, int iXcfBin, int iNthGpu);
	float DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam
	);
private:
	float mSearch(int iNumSteps, float fStep, float fInitOffset);
	float mCalcAveragedCC(float fTiltOffset);
	float mCorrelate(int iRefProj, int iProj);
	//----------------------------------------
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
	StreAlign::CStretchCC2D m_aStretchCC2D;
	Correct::CCorrTomoStack* m_pCorrTomoStack;
};

class CTiltAxisMain
{
public:
	CTiltAxisMain(void);
	~CTiltAxisMain(void);
	void Clean(void);
	void Setup
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam,
	  float fBFactor,
	  int iXcfBin
	);
	float Measure(float fTiltAxis);
private:
	float mCorrelate(int iProj, int iStep, float fTiltAxis);
	void mGetGProj(int iProj, float* gfPadProj);
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
	Util::GFFT2D m_fft2D;
	float m_fBFactor;
	int m_iXcfBin;
	int m_aiPadSize[2];
	float* m_gfPadProj1;
	float* m_gfPadProj2;
};
}
