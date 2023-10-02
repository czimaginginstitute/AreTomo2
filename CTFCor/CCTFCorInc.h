#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <Util/Util_Thread.h>
#include <cufft.h>

namespace CTFCor
{

struct CCTFParam
{
	float m_fWL; // normalized by pixel size
	float m_fCs; // normalized by pixel size
	float m_fDefocusMax; // normalized by pixel size
	float m_fDefocusMin; // normalized by pixel size
	float m_fPhaseShiftAmp; // radian
	float m_fPhaseShiftExt; // radian
	float m_fAstAzimuth; // astigmatism, radian
};

class GCalcCTF2D
{
public:
	GCalcCTF2D(void);
	~GCalcCTF2D(void);
	void Clean(void);
	void Setup
	( float fPixelSize, // Angstrom
	  float fKv,  // keV
	  float fCs,  // nm
	  float fAmpContrast,
	  float fPhaseShiftExt // radian
	);
	void SetSize(int iCmpX, int iCmpY);
	void DoIt
	( float fDefocusMin, // Angstrom
	  float fDefocusMax, // Angstrom
	  float fAstAzimuth  // radian
	);
	static float CalcWaveLength(float fKv);
	//-------------------------------------
	float* m_gfCTF;
	int m_aiCmpSize[2];
	float m_fFreq0; // Frequency at first zero
private:
	CCTFParam m_aCTFParam;
	float m_fPI;
	float m_fPixelSize;
};

class GCorCTF2D
{
public:
	GCorCTF2D(void);
	~GCorCTF2D(void);
	void Clean(void);
	void SetSize(int iCmpX, int iCmpY);
	void DoIt(cufftComplex* gCmpImg, float* gfCTF, float fFreq0);
private:
	float mCalcPower(float fStartFreq, float fEndFreq);
	float mSum1d(float* gfData, int iSize);
	float mSum1dCPU(float* gfData, int iSize);
	//----------------------------------------
	cufftComplex* m_gCmpImg;	
	float* m_gfSumPower;
	float* m_gfCount;
	dim3 m_aGridDim;
	dim3 m_aBlockDim;
	int m_aiCmpSize[2];
	float m_fPI;
};

class GDoseWeightImage
{
public:
	GDoseWeightImage(void);
	~GDoseWeightImage(void);
	void Clean(void);
	void BuildWeight(float fPixelSize, float fKv,
	   float* pfImgDose, int* piStkSize, 
	   cudaStream_t stream = 0);
	void DoIt(cufftComplex* gCmpImg, float fDose,
	   cudaStream_t stream = 0);
	float* m_gfWeightSum;
	int m_aiCmpSize[2];
};

class GCorBilinear
{
public:
	GCorBilinear(void);
	~GCorBilinear(void);
	void DoIt(cufftComplex* gCmpImg, int* piCmpSize);
};

class CCorrTomoStack : public Util_Thread
{
public:
	CCorrTomoStack(void);
	virtual ~CCorrTomoStack(void);
	static void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  float fKv,
	  float fCs,
	  float fPixelSize,
	  float fDefocus,
	  int* piGpuIDs,
	  int iNumGpus
	);
	void Clean(void);
	void Run(Util::CNextItem* pNextItem, int iGpuID);
	void ThreadMain(void);
private:
	void mCalcCTF(void);
	void mCorrectProj(int iProj);
	void mForwardFFT(int iProj);
	void mInverseFFT(int iProj);
	float* m_gfCTF;
	float m_fFreq0;
	cufftComplex* m_gCmpImg;
	Util::CNextItem* m_pNextItem;
	int m_iGpuID;
	int m_aiCmpSize[2];
	Util::CCufft2D m_aForwardFFT;
	Util::CCufft2D m_aInverseFFT;
	GCorCTF2D m_aGCorCTF2D;
	GCorBilinear m_aGCorBilinear;
};

class CWeightTomoStack : public Util_Thread
{
public:
	CWeightTomoStack(void);
	virtual ~CWeightTomoStack(void);
	static void DoIt(MrcUtil::CTomoStack* pTomoStack,
	   int* piGpuIDs, int iNumGpus);
	void Clean(void);
	void Run(Util::CNextItem* pNextItem, int iGpuID);
	void ThreadMain(void);
private:
	void mCorrectProj(int iProj);
	void mForwardFFT(int iProj);
	void mInverseFFT(int iProj);
	void mDoseWeight(int iProj);
	cufftComplex* m_gCmpImg;
	int m_aiCmpSize[2];
	Util::CCufft2D m_aForwardFFT;
	Util::CCufft2D m_aInverseFFT;
	Util::CNextItem* m_pNextItem;
	GDoseWeightImage* m_pGDoseWeightImg;
	int m_iGpuID;
};
}
