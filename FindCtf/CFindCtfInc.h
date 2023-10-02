#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <Util/Util_Thread.h>
#include <cufft.h>

namespace FindCtf
{

class CCtfParam
{
public:
	CCtfParam(void);
	~CCtfParam(void);
	float GetWavelength(bool bAngstrom);
	float GetDefocusMax(bool bAngstrom);
	float GetDefocusMin(bool bAngstrom);
	CCtfParam* GetCopy(void);
	float m_fWavelength; // pixel
	float m_fCs; // pixel
	float m_fAmpContrast;
	float m_fAmpPhaseShift; // radian
	float m_fExtPhase;   // radian
	float m_fDefocusMax; // pixel
	float m_fDefocusMin; // pixel
	float m_fAstAzimuth; // radian
	float m_fAstTol;     // Allowed astigmatism
	float m_fPixelSize;  // Angstrom
};

class CCtfTheory
{
public:
	CCtfTheory(void);
	~CCtfTheory(void);
	void Setup
	( float fKv, // keV
	  float fCs, // mm
	  float fAmpContrast,
	  float fPixelSize,    // A
	  float fAstTil,       // A, negative means no tolerance
	  float fExtPhase      // Radian
	);
	void SetExtPhase(float fExtPhase, bool bDegree);
	float GetExtPhase(bool bDegree);
	void SetPixelSize(float fPixSize);
	void SetDefocus
	( float fDefocusMin, // A
	  float fDefocusMax, // A
	  float fAstAzimuth  // deg
	);
	void SetDefocusInPixel
	( float fDefocusMaxPixel, // pixel
	  float fDefocusMinPixel, // pixel
	  float fAstAzimuthRadian // radian
	);
	void SetParam(CCtfParam* pCTFParam); // copy values
	CCtfParam* GetParam(bool bCopy);  // do not free
	float Evaluate
	( float fFreq, // relative frequency in [-0.5, +0.5]
	  float fAzimuth
	);
	int CalcNumExtrema
	( float fFreq, // relative frequency in [-0.5, +0.5]
	  float fAzimuth
	);
	float CalcNthZero
	( int iNthZero,
	  float fAzimuth
	);
	float CalcDefocus
	( float fAzimuth
	);
	float CalcPhaseShift
	( float fFreq, // relative frequency [-0.5, 0.5]
	  float fAzimuth
	);
	float CalcFrequency
	( float fPhaseShift,
	  float fAzimuth
	);
	bool EqualTo
	( CCtfTheory* pCTFTheory,
	  float fDfTol
	);
	float GetPixelSize(void);
	CCtfTheory* GetCopy(void);
private:
	float mCalcWavelength(float fKv);
	void mEnforce(void);
	CCtfParam* m_pCtfParam;
	float m_fPI;
};

class CCtfResults
{
public:
	static CCtfResults* GetInstance(void);
	static void DeleteInstance(void);
	~CCtfResults(void);
	void Clean(void);
	void Setup(int iNumImgs, int* piSpectSize);
	void SetTilt(int iImage, float fTilt);
	void SetDfMin(int iImage, float fDfMin);
	void SetDfMax(int iImage, float fDfMax);
	void SetAzimuth(int iImage, float fAzimuth);
	void SetExtPhase(int iImage, float fExtPhase);
	void SetScore(int iImage, float fScore);
	void SetSpect(int iImage, float* pfSpect);
	//----------------------------------------
	float GetTilt(int iImage);
	float GetDfMin(int iImage);
	float GetDfMax(int iImage);
	float GetAzimuth(int iImage);
	float GetExtPhase(int iImage);
	float GetScore(int iImage);
	float* GetSpect(int iImage, bool bClean);
	void SaveImod(void);
	void Display(int iNthCtf);
	void DisplayAll(void);
	//-----------------------
	int m_aiSpectSize[2];
	int m_iNumImgs;
private:
	CCtfResults(void);
	void mInit(void);
	float* m_pfDfMins;
	float* m_pfDfMaxs;
	float* m_pfAzimuths;
	float* m_pfExtPhases;
	float* m_pfScores;
	float* m_pfTilts;
	float** m_ppfSpects;
	static CCtfResults* m_pInstance;
};

class GCalcCTF1D
{
public:
	GCalcCTF1D(void);
	~GCalcCTF1D(void);
	void SetParam(CCtfParam* pCtfParam);
	void DoIt
	( float fDefocus,  // in pixel
	  float fExtPhase, // phase in radian from phase plate
	  float* gfCTF1D,
	  int iCmpSize
	);
private:
	float m_fAmpPhase;
};

class GCalcCTF2D
{
public:
	GCalcCTF2D(void);
	~GCalcCTF2D(void);
	void SetParam(CCtfParam* pCtfParam);
	void DoIt
	( float fDfMin, float fDfMax, float fAzimuth, 
	  float fExtPhase, // phase in radian from phase plate
	  float* gfCTF2D, int* piCmpSize
	);
	void DoIt
	( CCtfParam* pCtfParam,
	  float* gfCtf2D,
	  int* piCmpSize
	);
	void EmbedCtf
	( float* gfCtf2D,
	  float fMinFreq,
	  float fMaxFreq, // relative freq
	  float fMean, float fGain, // for scaling
	  float* gfFullSpect,
	  int* piCmpSize  // size of gfCtf2D
	); 

private:
	float m_fAmpPhase; // phase from amplitude contrast
};

class GCalcSpectrum
{
public:
	GCalcSpectrum(void);
	~GCalcSpectrum(void);
	void DoIt
	( cufftComplex* gCmp,
	  float* gfSpectrum,
	  int* piCmpSize,
	  bool bLog
	);
	void DoPad
	( float* gfPadImg,   // image already padded
	  float* gfSpectrum, // GPU buffer
	  int* piPadSize,
	  bool bLog
	);
	void Logrithm
	( float* gfSpectrum,
	  int* piSize
	);
	void GenFullSpect
	( float* gfHalfSpect,
	  int* piCmpSize,
	  float* gfFullSpect
	);
};

class GBackground1D
{
public:
	GBackground1D(void);
	~GBackground1D(void);
	void SetBackground(float* gfBackground, int iStart, int iSize);
	void Remove1D(float* gfSpectrum, int iSize);
	void Remove2D(float* gfSpectrum, int* piSize);
	void DoIt(float* pfSpectrum, int iSize);
	int m_iSize;
	int m_iStart;
private:
	int mFindStart(float* pfSpectrum);
	float* m_gfBackground;
};

class GRemoveMean
{
public:
	GRemoveMean(void);
	~GRemoveMean(void);
	void DoIt
	(  float* pfImg,  // 2D image
	   bool bGpu,     // if the image is in GPU memory
	   int* piImgSize // image x and y sizes
	);
	void DoPad
	(  float* pfPadImg, // 2D image with x dim padded
	   bool bGpu,       // if the image is in GPU memory
	   int* piPadSize   // x size is padded size
	);
private:
	float* mToDevice(float* pfImg, int* piSize);
	float mCalcMean(float* gfImg);
	void mRemoveMean(float* gfImg, float fMean);
	int m_iPadX;
	int m_aiImgSize[2];
};

class GRmBackground2D
{
public:
	GRmBackground2D(void);
	~GRmBackground2D(void);
	void DoIt
	( float* gfInSpect, // half spact
	  float* gfOutSpect,
	  int* piCmpSize,
	  float fMinFreq // relative frequency[0, 0.5]
	);
};

class GRadialAvg
{
public:
	GRadialAvg(void);
	~GRadialAvg(void);
	void DoIt(float* gfSpect, float* gfAverage, int* piCmpSize);
};

class GRoundEdge
{
public:
	GRoundEdge(void);
	~GRoundEdge(void);
	void SetMask
	(  float* pfCent,
	   float* pfSize
	);
	void DoIt
	(  float* gfImg,
	   int* piImgSize
	);

private:
	float m_afMaskCent[2];
	float m_afMaskSize[2];
};

class GCC2D
{
public:
	GCC2D(void);
	~GCC2D(void);
	void Setup
	(  float fFreqLow,  // relative freq [0, 0.5]
	   float fFreqHigh, // relative freq [0, 0.5]
	   float fBFactor
	);
	void SetSize(int* piCmpSize); // half spectrum
	float DoIt(float* gfCTF, float* gfSpectrum);
private:
	float m_fFreqLow;
	float m_fFreqHigh;
	float m_fBFactor;
	int m_aiCmpSize[2];
	int m_iGridDimX;
	int m_iBlockDimX;
	float* m_gfRes;
};

class GCC1D
{
public:
	GCC1D(void);
	~GCC1D(void);
	void SetSize(int iSize);
	void Setup
	(  float fFreqLow,   // relative freq [0, 0.5]
	   float fFreqHigh,  // relative freq [0, 0.5]
	   float fBFactor
	);
	float DoIt(float* gfCTF, float* gfSpectrum);
	float DoCPU
	(  float* gfCTF,
	   float* gfSpectrum,
	   int iSize
	);
private:
	int m_iSize;
	float* m_gfRes;
	float m_fFreqLow;
	float m_fFreqHigh;
	float m_fBFactor;
};


class CGenAvgSpectrum
{
public:
	CGenAvgSpectrum(void);
	~CGenAvgSpectrum(void);
	void Clean(void);
	void SetSizes(int* piImgSize,int iTileSize);
	void DoIt(float* pfImage, float* gfAvgSpect);
	int m_aiCmpSize[2];
private:
	void mGenAvgSpectrum(void);
	void mCalcTileSpectrum(int iTile);
	void mExtractPadTile(int iTile);
	Util::GCalcMoment2D* m_pGCalcMoment2D;
	float* m_pfImage;
	int m_aiImgSize[2];
	int m_iTileSize;
	int m_aiPadSize[2];
	int m_aiNumTiles[2];
	int m_aiOffset[2];
	int m_iOverlap;
	float m_fOverlap;
	float* m_gfAvgSpect;
	float* m_gfTileSpect;
	float* m_gfPadTile;
};

class CCalcBackground
{
public:
	CCalcBackground(void);
	~CCalcBackground(void);
	float* GetBackground(bool bClean);
	void DoSpline(float* pfSpectrum, int iSize, float fPixelSize);
	int m_iSize;
	int m_iStart;
private:
	int mFindStart(float* pfSpectrum);
	float* mLinearFit(float* pfData, int iStart, int iEnd);
	float* m_gfBackground;
	float m_fPixelSize;
};

class CSpectrumImage
{
public:
	CSpectrumImage(void);
	~CSpectrumImage(void);
     	void DoIt
	( float* gfHalfSpect,
	  float* gfCtfBuf,
	  int* piCmpSize,
	  CCtfTheory* pCtfTheory,
	  float* pfResRange,
	  float* gfFullSpect
	);
private:
	void mGenFullSpectrum(void);
	void mEmbedCTF(void);
	float* m_gfHalfSpect;
	float* m_gfCtfBuf;
	float* m_gfFullSpect;
	CCtfTheory* m_pCtfTheory;
	int m_aiCmpSize[2];
	float m_afResRange[2];
	float m_fMean;
	float m_fStd;     
};

class CFindDefocus1D
{
public:
	CFindDefocus1D(void);
	~CFindDefocus1D(void);
	void Clean(void);
	void Setup(CCtfParam* pCtfParam, int iCmpSize);
	void SetResRange(float afRange[2]); // angstrom
	void DoIt
	( float afDfRange[2],    // f0, delta angstrom
	  float afPhaseRange[2], // p0, delta degree
	  float* gfRadiaAvg
	);
	float m_fBestDf;
	float m_fBestPhase;
	float m_fMaxCC;
private:
	void mBrutalForceSearch(float afResult[3]);
	void mCalcCTF(float fDefocus, float fExtPhase);
	float mCorrelate(void);
	CCtfParam* m_pCtfParam;
	GCC1D* m_pGCC1D;
	GCalcCTF1D m_aGCalcCtf1D;
	float m_afResRange[2];
	float m_afDfRange[2];    // f0, delta in angstrom
	float m_afPhaseRange[2]; // p0, delta in degree
	float* m_gfRadialAvg;
	int m_iCmpSize;
	float* m_gfCtf1D;
};

class CFindDefocus2D 
{
public:
	CFindDefocus2D(void);
	~CFindDefocus2D(void);
	void Clean(void);
	void Setup1(CCtfParam* pCtfParam, int* piCmpSize);
	void Setup2(float afResRange[2]); // angstrom
	void Setup3
	( float fDfMean, float fAstRatio, 
	  float fAstAngle, float fExtPhase
	);
	void DoIt(float* gfSpect, float fPhaseRange);
	void Refine
	( float* gfSpect, float fDfMeanRange,
	  float fAstRange, float fAngRange,
	  float fPhaseRange
	);
	float GetDfMin(void);    // angstrom
	float GetDfMax(void);    // angstrom
	float GetAstRatio(void);
	float GetAngle(void);    // degree
	float GetExtPhase(void); // degree
	float GetScore(void);
private:
	float mIterate
	( float afAstRanges[2], float fDfMeanRange,
	  float fPhaseRange, int iIterations
	);
	float mGridSearch(float fRatRange, float fAngRange);
	float mRefinePhase(float fPhaseRange);
	float mRefineDfMean(float fRatRange);
	float mCorrelate(float fAzimu, float fAstig, float fExtPhase);
	float* m_gfSpect;
	float* m_gfCtf2D;
	int m_aiCmpSize[2];
	GCC2D* m_pGCC2D;
	GCalcCTF2D m_aGCalcCtf2D;
	CCtfParam* m_pCtfParam;
	float m_fDfMean;
	float m_fAstRatio;
	float m_fAstAngle;
	float m_fExtPhase;
	float m_fCCMax;
};

class CFindCtfBase
{
public:
	CFindCtfBase(void);
	virtual ~CFindCtfBase(void);
	void Clean(void);
	void Setup1(CCtfTheory* pCtfTheory);
	void Setup2(int* piImgSize);
	void SetPhase(float fInitPhase, float fPhaseRange); // degree
	void SetHalfSpect(float* pfCtfSpect);
	float* GetHalfSpect(bool bRaw, bool bToHost);
	void GetSpectSize(int* piSize, bool bHalf);
	void GenHalfSpectrum(float* pfImage);
	float* GenFullSpectrum(void); // clean by caller
	void ShowResult(void);
	float m_fDfMin;
	float m_fDfMax;
	float m_fAstAng;   // degree
	float m_fExtPhase; // degree
	float m_fScore;
protected:
	void mRemoveBackground(void);
	void mInitPointers(void);
	CCtfTheory* m_pCtfTheory;
	CGenAvgSpectrum* m_pGenAvgSpect;
	float* m_gfFullSpect;
	float* m_gfRawSpect;
	float* m_gfCtfSpect;
	int m_aiCmpSize[2];
	int m_aiImgSize[2];
	float m_afResRange[2];
	float m_fPhaseRange; // for searching extra phase in degree
};

class CFindCtf1D : public CFindCtfBase
{
public:
	CFindCtf1D(void);
	virtual ~CFindCtf1D(void);
	void Clean(void);
	void Setup1(CCtfTheory* pCtfTheory);
	void Do1D(void);
	void Refine1D(float fInitDf, float fDfRange);
protected:
	void mFindDefocus(void);
	void mRefineDefocus(float fDfRange);
	void mCalcRadialAverage(void);
	CFindDefocus1D* m_pFindDefocus1D;
	float* m_gfRadialAvg;
};

class CFindCtf2D : public CFindCtf1D
{
public:
	CFindCtf2D(void);
	virtual ~CFindCtf2D(void);
	void Clean(void);
	void Setup1(CCtfTheory* pCtfTheory);
	void Do2D(void);
	void Refine
	( float afDfMean[2], 
	  float afAstRatio[2],
	  float afAstAngle[2],
	  float afExtPhase[2]
	);
private:
	void mGetResults(void);
	CFindDefocus2D* m_pFindDefocus2D;
};

class CFindCtfHelp
{
public:
	static float CalcAstRatio(float fDfMin, float fDfMax);
	static float CalcDfMin(float fDfMean, float fAstRatio);
	static float CalcDfMax(float fDfMean, float fAstRatio);
};

class CSaveCtfResults : public Util_Thread
{
public:
	static CSaveCtfResults* GetInstance(void);
	static void DeleteInstance(void);
	~CSaveCtfResults(void);
	void AsyncSave(void);
	void ThreadMain(void);
private:
	CSaveCtfResults(void);
	void mSaveImages(void);
	void mSaveFittings(void);
	char m_acInMrcFile[256];
	char m_acOutFolder[256];
	static CSaveCtfResults* m_pInstance;
};

class CFindCtfMain
{
public:
	CFindCtfMain(void);
	~CFindCtfMain(void);
	void Clean(void);
	bool CheckInput(void);
	void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam
	);
private:
	void mGenSpectrums(void);
	void mDoZeroTilt(void);
	void mDo2D(void);
	void mSaveSpectFile(void);
	void mGetResults(int iTilt);
	char* mGenSpectFileName(void);
	float** m_ppfHalfSpects;
	CFindCtf2D* m_pFindCtf2D;
	int m_iNumTilts;
	int m_iRefTilt;
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
};

}
