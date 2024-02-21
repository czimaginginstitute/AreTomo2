#pragma once
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <Util/Util_Thread.h>
#include <cufft.h>

namespace StreAlign
{
class CStretchXcf
{
public:
	CStretchXcf(void);
	~CStretchXcf(void);
	void Clean(void);
	void Setup(int* piImgSize, float fBFactor);
	void DoIt
	( float* pfRefImg, float* pfImg,
	  float fRefTilt, float fTilt, float fTiltAxis
	);
	void GetShift(float fFactX, float fFactY, float* pfShift);
private:
	void mPadImage(float* pfImg, float* gfPadImg);
	void mNormalize(float* gfPadImg);
	void mRoundEdge(void);
	void mForwardFFT(void);
	cufftComplex* m_gCmpRef;
	cufftComplex* m_gCmpBuf;
	cufftComplex* m_gCmp;
	float m_fBFactor;
	Util::GFFT2D m_fft2D;
	Util::GXcf2D m_xcf2D;
	int m_aiImgSize[2];
	int m_aiPadSize[2];
	int m_aiCmpSize[2];
	float m_afShift[2];
};

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
	  float fTilt,
	  float fTiltAxis
	);
private:
	int m_aiSize[2];
	bool m_bPadded;
	float* m_gfRefImg;
	float* m_gfImg;
	float* m_gfBuf;
};

class CStretchAlign : public Util_Thread
{
public:
	CStretchAlign(void);
	~CStretchAlign(void);
	static float DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam,
	  float fBFactor,
	  float* pfBinning,
	  int* piGpuIDs,
	  int iNumGpus
	);
	void Run(Util::CNextItem* pNextItem, int iGpuID);
	void ThreadMain(void);
	float m_fMaxErr;
private:
	float mMeasure(int iProj);
	int mFindRefIndex(int iProj);
	Util::CNextItem* m_pNextItem;
	int m_iGpuID;
	CStretchXcf m_stretchXcf;
};

class CStreAlignMain
{
public:
	CStreAlignMain(void);
	~CStreAlignMain(void);
	void Clean(void);
	void Setup
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam
	);
	void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam
	);
private:
	float mMeasure(void);
	void mUpdateShift(void);
	void mUnstretch
	( int iLowTilt, int iHighTilt, 
	  float* pfShift
	);
	Correct::CCorrTomoStack* m_pCorrTomoStack;
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
	MrcUtil::CTomoStack* m_pBinStack;
	MrcUtil::CAlignParam* m_pMeaParam;
	float m_afBinning[2];
};

}
