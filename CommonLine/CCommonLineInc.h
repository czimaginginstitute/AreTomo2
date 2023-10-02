#pragma once
#include <Util/Util_Thread.h>
#include <Util/Util_Powell.h>
#include <CuUtil/DeviceArray2D.h>
#include <CuUtilFFT/GFFT1D.h>
#include <cufft.h>
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"

namespace CommonLine
{

class CCommonLineParam
{
public:
	static CCommonLineParam* GetInstance(void);
	static void DeleteInstance(void);
	~CCommonLineParam(void);
	void Clean(void);
	void Setup
	( float fAngRange, 
	  int iNumSteps,
	  MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam
	);
	int m_fAngRange;
	int m_iNumLines;
	float m_fTiltAxis;
	float* m_pfRotAngles;
	int m_iLineSize;
	int m_iCmpLineSize;
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
private:
	CCommonLineParam(void);
	static CCommonLineParam* m_pInstance;
};
	
class CPossibleLines
{
public:
	CPossibleLines(void);
	~CPossibleLines(void);
	void Clean(void);
	void Setup(void);
	void SetPlane
	( int iProj,
	  cufftComplex* gCmpPlane
	);
	void SetLine
	( int iProj, 
	  int iLine, 
	  cufftComplex* gCmpLine
	);
	void GetLine
	( int iProj, 
	  int iLine, 
	  cufftComplex* gCmpLine
	);
	float CalcLinePos(float fRotAngle);
	float GetLineAngle(int iLine);
	//----------------------------
	cufftComplex** m_ppCmpPlanes;
	float* m_pfRotAngles;
	float* m_pfTiltAngles;
	int m_iNumProjs;
	int m_iNumLines;
	int m_iCmpSize;
	int m_iLineSize;
private:
	cufftComplex* mGetLine(int iProj, int iLine);
};

class CLineSet
{
public:
	CLineSet(void);
	~CLineSet(void);
	void Clean(void);
	void Setup(void);
	cufftComplex* GetLine(int iLine); // do not free
	int GetLineGpu(int iLine);
	int GetGpuID(int iNthGpu);
	//------------------------
	int m_iNumProjs;
	int m_iCmpSize;
	int m_iNumGpus;
private:
	cufftComplex** m_pgCmpLines;
	int* m_piLineGpus;
	int* m_piGpuIDs;
};

class CSumLines : public Util_Thread
{
public:
	static cufftComplex* DoIt(CLineSet* pLineSet);
	CSumLines(void);
	virtual ~CSumLines(void);
	void Clean(void);
	cufftComplex* GetSum(bool bClean);
	void Run(int iThreadID);
	void ThreadMain(void);
private:
	int m_iGpuID;
	int m_iCmpSize;
	cufftComplex* m_gCmpSum;
};

class GInterpolateLineSet : public Util_Thread
{
public:
	static void DoIt
	( CPossibleLines* pPossibleLines,
	  float* pfRotAngles,
	  CLineSet* pLineSet
	);
	GInterpolateLineSet(void);
	~GInterpolateLineSet(void);
	void Clean(void);
	void Run(int iThreadID);
	void ThreadMain(void);
private:
	void mInterpolate(int iProj);
	cufftComplex* m_gCmpLine1;
	cufftComplex* m_gCmpLine2;
	int m_iCmpSize;
	int m_iGpuID;
};	

class GCC1D
{
public:
	GCC1D(void);
	~GCC1D(void);
	void SetLowpass(float fBFactor);
	float DoIt
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  int iCmpSize
	);
private:
	float mTestOnCPU
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  int iCmpSize
	);
	float m_fBFactor;
};

class GCalcCommonRegion
{
public:
	GCalcCommonRegion(void);
	~GCalcCommonRegion(void);
	void Clean(void);
	void DoIt(void);
	int* m_giComRegion;
};

class GGenCommonLine
{
public:
	GGenCommonLine(void);
	~GGenCommonLine(void);
	void Clean(void);
	void Setup(void);
	void DoIt
	( int iProj,
	  int* giComRegion,
	  float* gfPadLines // padded buffer for common line
	);
private:
	float* m_gfImage;
	float* m_gfRotAngles;
	int m_aiImgSize[2];
	int m_iNumAngles;
};

class GRemoveMean
{
public:
	GRemoveMean(void);
	~GRemoveMean(void);
	void DoIt(float* gfPadLine, int iPadSize);
};

//--------------------------------------------------------------------
//
//--------------------------------------------------------------------
class GSumLines
{
public:
	GSumLines(void);
	~GSumLines(void);
	void SetLines
	(  cufftComplex* gCmpLines,
	   int iNumLines,
	   int iCmpSize
	);
	cufftComplex* DoIt
	(  int iExcludedLine
	);
private:
	cufftComplex* m_pCmpSum;
	cufftComplex* m_pCmpLines;
	int m_iNumLines;
	int m_iCmpSize;
};

class GFunctions
{
public:
	GFunctions(void);
	~GFunctions(void);
	void Sum
	( float* gfData1,
	  float* gfData2,
	  float fFact1,
	  float fFact2,
	  float* gfSum,
	  int iSize
	);
	void Sum
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  float fFact1,
	  float fFact2,
	  cufftComplex* gSum,
	  int iCmpSize
	);
};


//--------------------------------------------------------------------
//--------------------------------------------------------------------
class CGenLines : public Util_Thread
{
public:
	static CPossibleLines* DoIt(void);
	CGenLines(void);
	virtual ~CGenLines(void);
	void Run(int iThreadID, Util::CNextItem* pNextItem);
	void ThreadMain(void);
private:
	void mCalcCommonRegion(void);
	void mGenLines(void);
	void mGenProjLines(int iProj);
	void mForwardFFT(void);
	void mClean(void);
	//----------------
	cufftComplex* m_gCmpPlane;
	int m_iNumLines;
	int m_iLineSize;
	int m_iGpuID;
	//-----------
	GCalcCommonRegion m_calcComRegion;
	GGenCommonLine m_genComLine;
	CuUtilFFT::GFFT1D m_fft1D;
	//------------------------
	Util::CNextItem* m_pNextItem;
};

//--------------------------------------------------------------------
// 1. This class determines the coherence within a set of lines
//    obtained from each tilted image.
// 2. The coherence is the sum of all the correlation coefficients
//    of all the pairs of lines.
//--------------------------------------------------------------------
class GCoherence
{
public:
	GCoherence(void);
	~GCoherence(void);
	float DoIt
	(  cufftComplex* gCmpLines,
	   int iCmpSize,
	   int iNumLines
	);
private:
	void mCalcSum(void);
	float mMeasure
	(  int iLine
	);
	int m_iCmpSize;
	int m_iNumLines;
	cufftComplex* m_gCmpLines;
	cufftComplex* m_gCmpRef;
	cufftComplex* m_gCmpSum;
	float m_fCC;
};

class CCalcScore : public Util_Thread
{
public:
	static float DoIt
	( CLineSet* pLineSet,
	  cufftComplex* gCmpSum
	);
	CCalcScore(void);
	virtual ~CCalcScore(void);
	void Run(int iThreadID);
	void ThreadMain(void);
	//--------------------
	float m_fCCSum;
private:
	float mCorrelate(int iLine);
	cufftComplex* mCudaMallocLine(bool bZero);
	//----------------------------------------
	int m_iThreadID;
	int m_iGpuID;
	cufftComplex* m_gCmpSum;
	cufftComplex* m_gCmpRef;
	int m_iCmpSize;
};

class CFindTiltAxis
{
public:
	CFindTiltAxis(void);
	~CFindTiltAxis(void);
	void Clean(void);
	float DoIt
	( CPossibleLines* pPossibleLines,
	  CLineSet* pLineSet
	);
	float m_fScore;
private:
	int mDoIt(void);
	void mFillLineSet(int iLine);
	float mCalcScore(void);
	CPossibleLines* m_pPossibleLines;
	CLineSet* m_pLineSet;
	int m_iNumImgs;
	int m_iNumLines;
};
	

//--------------------------------------------------------------------
//
//--------------------------------------------------------------------
class CRefineTiltAxis : public Util_Powell
{
public:
	CRefineTiltAxis(void);
	virtual ~CRefineTiltAxis(void);
	void Clean(void);
	void Setup(int iDim, int iIterations, float fTol);
	float Refine	
	( CPossibleLines* pPossibleLines,
	  CLineSet* pLineSet
	);
	void GetRotAngles(float* pfRotAngles);
	float Eval(float* pfCoeff);
private:
	void mCalcRotAngles(float* pfCoeff);
	//----------------------------------
	CPossibleLines* m_pPossibleLines;
	CLineSet* m_pLineSet;
	//-------------------
	float* m_pfCoeff;
	float* m_pfTerms;
	float* m_pfSearchRange;
	float* m_pfRotAngles;
	//-------------------
	int m_iNumProjs;
	int m_iNumLines;
	float m_fRefTilt;
	float m_fRefRot;
	float m_fNormTilt;
};

//--------------------------------------------------------------------
//
//--------------------------------------------------------------------
class CCommonLineMain
{
public:
	CCommonLineMain(void);
	~CCommonLineMain(void);
	void Clean(void);
	float DoInitial
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam,
	  float fAngRange, //search range centered at current tilt axis 
	  int iNumSteps    //number of steps in the search range
	);
	float DoRefine
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam
	);
private:
	void mSmooth(void);
	bool mFit3
	( float* pfX, float* pfRots, float* pfW, 
	  int iSize, float afFit[4]
	);
	float* m_pfRotAngles;
	float* m_pfFitAngles;
	int m_iNumImgs;
};

}
