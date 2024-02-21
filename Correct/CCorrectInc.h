#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <Util/Util_Thread.h>
#include <cufft.h>

namespace Correct
{
class CCorrectUtil
{
public:
	static void CalcAlignedSize
	( int* piRawSize, float fTiltAxis, 
	  int* piAlnSize
	);
	static void CalcBinnedSize
	( int* piRawSize, float fBinning, bool bFourierCrop,
	  int* piBinnedSize
	);
};

class CBinStack
{
public:
	CBinStack(void);
	~CBinStack(void);
	void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  int iBinning,
	  MrcUtil::CTomoStack* pBinStack
	);
};

class CFourierCropImage
{
public:
	CFourierCropImage(void);
	~CFourierCropImage(void);
	void Setup(int iNthGpu, int* piImgSize, float fBin);
	void DoPad(float* gfPadImgIn, float* gfImgPadOut);
	int m_aiImgSizeIn[2];
	int m_aiImgSizeOut[2];
private:
	Util::GFFT2D* m_gForward2D;
	Util::GFFT2D* m_gInverse2D;
};

class CCorrLinearInterp
{
public:
	CCorrLinearInterp(void);
	~CCorrLinearInterp(void);
	void Setup(int* piImgSize);
	void DoIt(float* gfImg, float* gfPadBuf);
private:
	int m_aiImgSize[2];
	int m_aiCmpSize[2];
	Util::GFFT2D* m_gForward2D;
	Util::GFFT2D* m_gInverse2D;
};

class GCorrPatchShift
{
public:
	GCorrPatchShift(void);
	~GCorrPatchShift(void);
	void SetSizes
	( int* piInSize,
	  bool bInPadded,
	  int* piOutSize,
	  bool bOutPadded,
	  int iNumPatches
	);
	void DoIt
	( float* gfInImg,
	  float* pfGlobalShift,
	  float fRotAngle,
	  float* gfLocalAlnParams,
	  bool bRandomFill,
	  float* gfOutImg
	);
private:
	float m_fD2R;
	int m_iInImgX;
	int m_iOutImgX;
	int m_iOutImgY;
};

class CCorrProj
{
public:
	CCorrProj(void);
	~CCorrProj(void);
	void Clean(void);
	void Setup
	( int* piInSize, bool bInPadded,
	  bool bRandomFill, bool bFourierCrop,
	  float fTiltAxis, float fBinning,
	  int iNthGpu
	);
	void SetProj(float* pfInProj);
	void DoIt(float* pfGlobalShift, float fTiltAxis);
	void GetProj(float* pfCorProj, int* piSize, bool bPadded);
private:
	int m_aiInSize[2];
	int m_iInImgX;
	bool m_bInPadded;
	bool m_bRandomFill;
	bool m_bFourierCrop;
	float m_fBinning;
	int m_iNthGpu;
	//------------
	float* m_gfRawProj;
	float* m_gfCorProj;
	float* m_gfRetProj;
	int m_aiCorSize[2];
	int m_aiRetSize[2];
	//-----------------
	Util::GBinImage2D m_aGBinImg2D;
	CFourierCropImage m_aFFTCropImg;
	GCorrPatchShift m_aGCorrPatchShift;
};

class CCorrTomoStack 
{
public:
	CCorrTomoStack(void);
	~CCorrTomoStack(void);
	//-----------------------------------------------------------
	// In case of shift only, fTiltAxis must be zero.
	//-----------------------------------------------------------
	void Set0(int iGpuID);
	void Set1(int* piStkSize, int iNumPatches, float fTiltAxis);
	void Set2(float fOutBin, bool bFourierCrop, bool bRandFill);
	void Set3(bool bShiftOnly, bool bCorrInt, bool bRWeight);
	void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pFullParam,
	  MrcUtil::CLocalAlignParam* pLocalParam
	);
	MrcUtil::CTomoStack* GetCorrectedStack(bool bClean);
	void GetBinning(float* pfBinning);
	void Clean(void);
private:
	void mCorrectProj(int iProj);
	float* m_gfRawProj;
	float* m_gfCorrProj;
	float* m_gfBinProj;
	float* m_gfLocalParam;
	GCorrPatchShift m_aGCorrPatchShift;
	Util::GBinImage2D m_aGBinImg2D;
	CFourierCropImage m_aFFTCropImg;
	MrcUtil::CTomoStack* m_pOutStack;
	void* m_pGRWeight;
	CCorrLinearInterp* m_pCorrInt;
	//-------------------------------
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pFullParam;
	MrcUtil::CLocalAlignParam* m_pLocalParam;
	float m_fOutBin;
	float m_afBinning[2];
	int m_aiStkSize[3];
	int m_aiAlnSize[3];
	int m_aiBinnedSize[3];
	bool m_bShiftOnly;
	bool m_bRandomFill;
	bool m_bFourierCrop;
	int m_iGpuID;
};

}
