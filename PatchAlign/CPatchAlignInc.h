#pragma once
#include "../MrcUtil/CMrcUtilInc.h"
#include "../ProjAlign/CProjAlignInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <Util/Util_Thread.h>
#include <cuda.h>
#include <cufft.h>
#include <pthread.h>

namespace PatchAlign
{
class GAddImages
{
public:
	GAddImages(void);
	~GAddImages(void);
	void DoIt
	( float* gfImg1, float fFactor1,
	  float* gfImg2, float fFactor2,
	  float* gfSum, int* piImgSize,
	  cudaStream_t stream = 0
	);
};

class GRandom2D
{
public:
	GRandom2D(void);
	~GRandom2D(void);
	void DoIt
	( float* gfInImg,
	  float* gfOutImg,
	  int* piImgSize,
	  bool bPadded,
	  cudaStream_t stream = 0
	);
};

class GExtractPatch
{
public:
	GExtractPatch(void);
	~GExtractPatch(void);
	void SetSizes
	( int* piInSize, bool bInPadded,
	  int* piOutSize, bool bOutPadded
	);
	void DoIt
	( float* gfInImg, int* piShift,
	  bool bRandomFill, float* gfOutImg
	);
private:
	int m_iInImgX;
	int m_iOutImgX;
	int m_aiOutSize[2];
};

class GCommonArea
{
public:
	GCommonArea(void);
	~GCommonArea(void);
	void DoIt
	( float* gfImg1,
	  float* gfImg2,
	  float* gf2Buf2,
	  int* piImgSize,
	  bool bPadded,
	  float* gfCommArea,
	  cudaStream_t stream = 0
	);
private:
	void mFindCommonArea(void);
	void mCenterCommonArea(float* gfCommArea);
	int m_aiImgSize[2];
	int m_iPadX;
	float* m_gfImg1;
	float* m_gfImg2;
	float* m_gfBuf1;
	float* m_gfBuf2;
	cudaStream_t m_stream;
};

class GGenXcfImage
{
public:
	GGenXcfImage(void);
	~GGenXcfImage(void);
	void Clean(void);
	void Setup(int* piCmpSize);
	void DoIt
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  float* gfXcfImg,
	  float fBFactor,
	  cudaStream_t stream = 0
	);
	int m_aiXcfSize[2];
private:
	CuUtilFFT::GFFT2D m_fft2D;
};

class GPartialCopy
{
public:
	GPartialCopy(void);
	~GPartialCopy(void);
	void DoIt
	( float* gfSrc, int iSrcSizeX,
	  float* gfDst, int* piDstSize,
	  int iCpySizeX,
	  cudaStream_t stream = 0
	);
	void DoIt
	( float* gfSrcImg, int* piSrcSize, int* piSrcStart,
	  float* gfPatch, int* piPatSize, bool bPadded,
	  cudaStream_t stream = 0
	);
};

class GNormByStd2D
{
public:
	GNormByStd2D(void);
	~GNormByStd2D(void);
	void DoIt(float* gfImg, int* piImgSize, bool bPadded,
	   int* piWinSize, cudaStream_t stream = 0);
};

class CCalcXcfImage
{
public:
	CCalcXcfImage(void);
	~CCalcXcfImage(void);
	void Clean(void);
	void Setup(int* piPatSize); 
	void DoIt
	( float* gfImg1,
	  float* gfImg2,
	  float* gfBuf,
	  int* piImgSize,
	  int* piStart,
	  float* gfXcfImg,
	  int* piXcfSize,
	  float* gfCommArea, 
	  cudaStream_t stream = 0
	);
private:
	void mExtractPatches
	( float* gfImg1, float* gfImg2, 
	  int* piImgSize, int* piStart
	);
	void mNormalize(float* gfPatImg);
	void mRoundEdge(float* gfPatImg);
	//-------------------------------
	int m_aiPatSize[2];
	int m_aiPatPad[2]; // patch padded size
	GGenXcfImage m_GGenXcfImage;
	CuUtilFFT::GFFT2D m_Gfft2D;
	float* m_gfPatImg1;
	float* m_gfPatImg2;
	float* m_gfBuf;
	cudaStream_t m_stream;
};

class CFitPatchShifts
{
public:
	CFitPatchShifts(void);
	~CFitPatchShifts(void);
	void Clean(void);
	void Setup
	( MrcUtil::CAlignParam* pFullParam,
	  int iNumPatches
	);
	float DoIt
	( MrcUtil::CPatchShifts* pPatchShifts,
	  MrcUtil::CLocalAlignParam* pLocalAlnParam
	);
	int m_iNumPatches;
	int m_iNumTilts;
private:
	void mCalcPatCenters(void);
	//void mCalcXs(void);
	float mCalcZs(void);
	float mCalcPatchZ(int iPatch);
	float mRefineTiltAxis(void);
	float mCalcTiltAxis(float fDelta);
	void mCalcSinCosRots(void);
	void mCalcLocalShifts(void);
	void mCalcPatchLocalShifts(int iPatch);
	void mScreenPatchLocalShifts(int iPatch);
	void mScreenTiltLocalShifts(int iTilt);
	//-------------------------------------
	MrcUtil::CPatchShifts* m_pPatchShifts;
	MrcUtil::CAlignParam* m_pFullParam;
	MrcUtil::CLocalAlignParam* m_pLocalParam;
	float* m_pfMeasuredUs;
	float* m_pfMeasuredVs;
	float* m_pfPatCentUs;
	float* m_pfPatCentVs;
	float* m_pfCosTilts;
	float* m_pfSinTilts;
	float* m_pfCosRots;
	float* m_pfSinRots;
	float* m_pfDeltaRots;
	//float* m_pfXs;
	float m_fErr;
	int m_iZeroTilt;
};

class CLocalMeasure
{
public:
	CLocalMeasure(void);
	~CLocalMeasure(void);
	void Clean(void);
	void Setup
	( int* piImgSize, bool bPadded, int* piNumPatches, 
	  MrcUtil::CPatchShifts* pPatchShifts
	);
	void DoIt(int iImage, float* pfReproj, float* pfProj);
private:
	void mGenXcfImg(int iPatch);
	void mGetPatchStart(int iPatch, int* piStart);
	void mGetPatchCenter(int iPatch, float* pfCenter);
	//------------------------------------------------
	int m_aiImgSize[2];
	int m_iImgPadX;
	int m_aiNumPatches[2];
	int m_iNumPatches;
	int m_aiPatSize[2];
	int m_aiXcfSize[2];
	int m_iImage;
	//-----------
	float* m_gfBuf;
	float* m_gfReproj;
	float* m_gfProj;
	float* m_gfImgBuf;
	float* m_gfXcfImg;
	float* m_gfCommAreas;
	float* m_pfXcfs;
	CCalcXcfImage m_aCalcXcfImage;
	MrcUtil::CPatchShifts* m_pPatchShifts;
};

class CExtTomoStack : public Util_Thread
{
public:
	static void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CTomoStack* pPatchStack,
	  int* piShifts,
	  bool bRandomFill,
	  int* piGpuIDs,
	  int iNumGpus
	);
	CExtTomoStack(void);
	~CExtTomoStack(void);
	void Clean(void);
	void Run(Util::CNextItem* pNextItem, int iGpuID);
	void ThreadMain(void);
private:
	void mExtractProj(int iProj);
	float* m_gfRawProj;
	float* m_gfPatProj;
	GExtractPatch m_aGExtractPatch;
	Util::CNextItem* m_pNextItem;
	int m_iGpuID;
};

class CLocalAlign
{
public:
	CLocalAlign(void);
	~CLocalAlign(void);
	void Setup
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam,
	  int iNthGpu
	);
	void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam,
	  int* piRoi
	);
private:
	int m_iZeroTilt;
	ProjAlign::CProjAlignMain* m_pProjAlignMain;
};

class CDetectFeatures
{
public:
	static CDetectFeatures* GetInstance(void);
	static void DeleteInstance(void);
	~CDetectFeatures(void);
	void SetSize(int* piImgSize, int* piNumPatches);
	void DoIt(float* pfImg);
	void GetCenter(int iPatch, int* piCent);
	
private:
	CDetectFeatures(void);
	void mClean(void);
	void mFindCenters(void);
	bool mCheckFeature(float fCentX, float fCentY);
	void mFindCenter(float* pfCenter);
	void mSetUsed(float fCentX, float fCentY);
	int mCheckRange(int iStartX, int iSize, int* piRange);
	int m_aiImgSize[2];
	int m_aiBinnedSize[2];
	int m_aiNumPatches[2];
	float m_afPatSize[2];
	int m_aiSeaRange[4];
	bool* m_pbFeatures;
	bool* m_pbUsed;
	float* m_pfCenters;
	static CDetectFeatures* m_pInstance;
};

class CRoiTargets
{
public:
	static CRoiTargets* GetInstance(void);
	static void DeleteInstance(void);
	CRoiTargets(void);
	~CRoiTargets(void);
	void LoadRoiFile();
	void SetTargetImage
	( MrcUtil::CAlignParam* pAlignParam
	);
	void MapToUntiltImage
	( MrcUtil::CAlignParam* pAlignParam,
	  MrcUtil::CTomoStack* pTomoStack
	);
	void GetTarget(int iTgt, int* piTgt);
	int* GetTargets(bool bClean);
	int m_iNumTgts;
private:
	int* m_piTargets;
	int m_iTgtImg;
	static CRoiTargets* m_pInstance;
};

class CPatchTargets
{
public:
	static CPatchTargets* GetInstance(void);
	static void DeleteInstance(void);
	CPatchTargets(void);
	~CPatchTargets(void);
	void Clean(void);
	void DetectTargets
	(  MrcUtil::CTomoStack* pTomoStack,
	   MrcUtil::CAlignParam* pAlignParam
	);
	void GetTarget(int iTgt, int* piTgt);
	int m_iNumTgts;
private:
	int* m_piTargets;
	int m_iTgtImg;
	static CPatchTargets* m_pInstance;
};

class CPatchAlignMain : Util_Thread
{
public:
	~CPatchAlignMain(void);

	static MrcUtil::CLocalAlignParam* DoIt
	( MrcUtil::CTomoStack* pTomoStack, 
	  MrcUtil::CAlignParam* pAlignParam,
	  float fTiltOffset
	);
	void Run(int iNthGpu);

	void ThreadMain(void);
private:
	CPatchAlignMain(void);

	void mAlignStack(int iPatch);
	//---------------------------
	CLocalAlign* m_pLocalAlign;
	int m_iNthGpu;
};
}
