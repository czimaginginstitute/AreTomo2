#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Recon/CReconInc.h"
#include "../Correct/CCorrectInc.h"
#include <Util/Util_Thread.h>
#include <cufft.h>

namespace ProjAlign
{
class CParam
{
public:
	static CParam* GetInstance(void);
	static void DeleteInstance(void);
	~CParam(void);
	float m_afMaskSize[2];
	int m_iIterations;
	float m_fTol;
	int m_iVolZ;
	float m_fXcfSize;
private:
	CParam(void);
	static CParam* m_pInstance;
};

class GReproj
{
public:
	GReproj(void);
	~GReproj(void);
	void Clean(void);
	void SetSizes(int iProjX, int iNumProjs, int iVolX, int iVolZ);
	void DoIt
	( float* gfSinogram,
	  float* gfTiltAngles,
	  int* piProjRange,
	  float fProjAngle,
	  cudaStream_t stream = 0
	);
	//-----------------------
	float* m_gfVol;
	float* m_gfReproj;
	int m_aiProjSize[2]; // proj size x and num projs
	cudaStream_t m_stream;
	
private:
	void mBackProj(float fProjAngle, int* piProjRange);
	void mForwardProj(float fProjAngle);
	float* m_gfSinogram;
	float* m_gfTiltAngles;
	int m_aiVolSize[2]; // vol size x and z
};

class CRemoveSpikes : public Util_Thread
{
public:
	static void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  int* piGpuIDs,
	  int iNumGpus
	);
	//------------
	CRemoveSpikes(void);
	~CRemoveSpikes(void);
	void Run(Util::CNextItem* pNextItem, int iGpuID);
	void ThreadMain(void);
private:
	Util::CNextItem* m_pNextItem;
	float* m_gfInFrm;
	float* m_gfOutFrm;
	size_t m_tFmBytes;
	int m_iGpuID;
};

class CCalcReproj 
{
public:
	CCalcReproj(void);
	~CCalcReproj(void);
	void Clean(void);
	void Setup(int* piTomoStackSize, int iVolZ, int iNthGpu);
	void DoIt
	( float** ppfProjs, float* pfTiltAngles, 
	  bool* pbSkipProjs, int iProjIdx, 
	  float* pfReproj // pinned
	);
private:
	void mFindProjRange(float* pfTiltRange, bool* pbSkipProjs);
	void mGetSinogram(int iY);
	void mReproj(int iY, float fProjAngle);
	void mAllocBuf(void);
	//-------------------
	float** m_ppfProjs;
	float* m_pfReproj;
	int m_iNthGpu;
	//-----------------
	int m_aiProjRange[2];
	int m_iProjIdx;
	int m_aiProjSize[2];
	int m_iNumProjs;
	float m_fMaxStch;
	float m_fMaxDiff;
	float* m_pfPinnedBuf;
	float* m_gfBuf;
	float* m_gfSinogram;
	float* m_gfTiltAngles;
	bool* m_gbSkipProjs;
	GReproj m_aGReproj;
};

class GProjXcf
{
public:
        GProjXcf(void);
        ~GProjXcf(void);
        void Clean(void);
        void Setup(int* piCmpSize);
        void DoIt
        ( cufftComplex* gCmp1,
          cufftComplex* gCmp2,
          float fBFactor,
	  float fPower
        );
        float SearchPeak(void);
        void GetShift
        ( float* pfShift,
          float fXcfBin
        );
        int m_aiXcfSize[2];
        float* m_pfXcfImg;
        float m_fPeak;
private:
        float m_afPeak[2];
        float m_fBFactor;
	float m_fPower;
        Util::GFFT2D m_fft2D;
};

class CCentralXcf
{
public:
	CCentralXcf(void);
	~CCentralXcf(void);
	void Clean(void);
	void Setup(int* piImgSize, int iVolZ);
	void SetupXcf(float fPower, float fBFactor);
	void DoIt(float* pfRef, float* pfImg, float fTilt);
	void GetShift(float* pfShift);
	float m_afShift[2];
private:
	void mGetCentral(float* pfImg, float* gfPadImg);
	void mNormalize(float* gfPadImg);
	void mCorrelate(void);
	//--------------------
	int m_aiImgSize[2];
	int m_iVolZ;
	float m_fTilt;
	float m_fPower;
	float m_fBFactor;
	//---------------
	int m_aiCentSize[2];
	int m_aiPadSize[2];
	int m_iXcfBin;
	//------------
	Util::GFFT2D m_fft2D;
	GProjXcf m_projXcf;
	float* m_gfPadRef;
	float* m_gfPadImg;
	float* m_gfPadBuf;
};

class CProjAlignMain
{
public:
	CProjAlignMain(void);
	~CProjAlignMain(void);
	void Clean(void);
	void Setup
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam,
	  float fBFactor, int iNthGpu
	);
	float DoIt(MrcUtil::CTomoStack* pTomoStack,
	   MrcUtil::CAlignParam* pAlignParam);
private:
	float mDoAll(void);
	void mDoPositive(void);
	void mDoNegative(void);
	float mAlignProj(int iProj);
	void mDoSubset(float fMinTilt, float fMaxTilt);
	float mMeasure(int iIter);
	void mCalcBinning(void);
	void mBinStack(void);
	void mRemoveSpikes(MrcUtil::CTomoStack* pTomoStack);
	void mCalcReproj(int iProj);
	void mCorrectProj(int iProj);
	//---------------------------
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
	Correct::CCorrTomoStack* m_pCorrTomoStack;
	//----------------------------------------
	float m_fBFactor;
	int m_iNthGpu;
	int m_iBin;
	int m_iVolZ;
	float m_afBinning[2];
	int m_iNumProjs;
	//--------------
	float* m_pfReproj;
	float* m_pfTiltAngles;
	bool* m_pbSkipProjs;
	CCalcReproj m_aCalcReproj;
	CCentralXcf m_centralXcf;
	Correct::CCorrProj m_corrProj;
	MrcUtil::CTomoStack* m_pBinStack;
	int m_iZeroTilt;
};

}
