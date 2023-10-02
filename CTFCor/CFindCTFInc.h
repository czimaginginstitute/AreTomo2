#pragma once
#include <cufft.h>

namespace GCTFFind
{

class CCTFParam
{
public:
	CCTFParam(void);
	~CCTFParam(void);
	float CalcWavelength(void); // Angstrom
	float m_fHT; // Volt
	float m_fCs; // Angstrom
	float m_fAmpContrast;
	float m_fAmpPhaseShift; // radian
	float m_fAddPhaseShift; // radian
	float m_fDefocusMax; // Angstrom
	float m_fDefocusMin; // Angstrom
	float m_fAstAzimuth; // radian
	float m_fPixelSize;  // Angstrom
};

class CCTFTheory
{
public:
	CCTFTheory(void);
	~CCTFTheory(void);
	void Setup
	(  float fKv, // keV
	   float fCs, // mm
	   float fAmpContrast,
	   float fPixelSize,    // A
	   float fAstTil,       // A, negative means no tolerance
	   float fAddPhaseShift // Radian
	);
	void SetDefocus
	(  float fDefocusMin, // A
	   float fDefocusMax, // A
	   float fAstAzimuth  // deg
	);
	void SetDefocusInPixel
	(  float fDefocusMaxPixel, // pixel
	   float fDefocusMinPixel, // pixel
	   float fAstAzimuthRadian // radian
	);
	void SetParam  // copy values
	(  CCTFParam* pCTFParam
	);
	CCTFParam* GetParam  // do not free
	(  void
	);
	float Evaluate
	(  float fFreq, // relative frequency in [-0.5, +0.5]
	   float fAzimuth
	);
	int CalcNumExtrema
	(  float fFreq, // relative frequency in [-0.5, +0.5]
	   float fAzimuth
	);
	float CalcNthZero
	(  int iNthZero,
	   float fAzimuth
	);
	float CalcDefocus
	(  float fAzimuth
	);
	float CalcPhaseShift
	(  float fFreq, // relative frequency [-0.5, 0.5]
	   float fAzimuth
	);
	float CalcFrequency
	(  float fPhaseShift,
	   float fAzimuth
	);
	bool EqualTo
	(  CCTFTheory* pCTFTheory,
	   float fDfTol
	);
	float GetPixelSize(void);
	CCTFTheory* GetCopy(void);
private:
	float mCalcWavelength(float fKv);
	void mEnforce(void);
	CCTFParam* m_pCTFParam;
	float m_fAstTol;
	float m_fPI;
};

class GCalcCTF1D
{
public:
	GCalcCTF1D(void);
	~GCalcCTF1D(void);
	void DoIt
	(  CCTFTheory* pCTFTheory,
	   float* gfCTF1D,
	   int iCmpSize
	);
};

class GCalcCTF2D
{
public:
	GCalcCTF2D(void);
	~GCalcCTF2D(void);
	void DoIt
	(  CCTFTheory* pCTFTheory,
	   float* gfCTF2D,
	   int* piCmpSize  //gfCTF2D size
	);
};

class GCalcSpectrum
{
public:
	GCalcSpectrum(void);
	~GCalcSpectrum(void);
	void DoIt
	(  cufftComplex* gCmp,
	   float* gfSpectrum,
	   int* piCmpSize
	);
	void DoPad
	(  float* gfPadImg,   // image already padded
	   float* gfSpectrum, // GPU buffer
	   int* piPadSize
	);
	void Logrithm
	(  float* gfSpectrum,
	   int* piSize
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

class GScaleToS2
{
public:
	GScaleToS2(void);
	~GScaleToS2(void);
	void Do2D
	(  float* gfSpectrum,
	   int* piSize
	);
	void Do1D
	(  float* gfSpectrum,
	   int iSize
	);
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
	(  float* gfSpectrum,
	   int* piCmpSize,
	   float fMinFreq // relative frequency[0, 0.5]
	);
};

class GRadialAvg
{
public:
	GRadialAvg(void);
	~GRadialAvg(void);
	float* DoIt
	(  float* gfSpectrum,
	   int* piCmpSize
	);
	int m_iSize;
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
	(  float fFreqLow,  // Fourier-space pixel
	   float fFreqHigh, // Fourier-space pixel
	   float fBFactor
	);
	float DoIt
	(  float* gfCTF,
	   float* gfSpectrum,
	   int* piCmpSize
	);
private:
	float m_fFreqLow;
	float m_fFreqHigh;
	float m_fBFactor;
};

class GCC1D
{
public:
	GCC1D(void);
	~GCC1D(void);
	void Setup
	(  float fFreqLow,   // Fourier-space pixel 
	   float fFreqHigh,  // Fourier-space pixel
	   float fBFactor
	);
	float DoIt
	(  float* gfCTF,
	   float* gfSpectrum,
	   int iSize
	);
	float DoCPU
	(  float* gfCTF,
	   float* gfSpectrum,
	   int iSize
	);
private:
	float m_fFreqLow;
	float m_fFreqHigh;
	float m_fBFactor;
};


class CGenAvgSpectrum
{
public:
	CGenAvgSpectrum(void);
	~CGenAvgSpectrum(void);
	void SetOverlap(float fOverlap);
	float* DoIt
	(  float* pfImage,
	   int* piImgSize,
	   int iTileSize
	);
	int m_aiCmpSize[2];
private:
	float* mGenAvgSpectrum(void);
	void mCalcTileSpectrum
	(  int iNthTile,
	   float* gfTileSpectrum
	);
	float* mExtractPadTile(int iNthTile);
	float* m_pfImage;
	int m_aiImgSize[2];
	int m_iTileSize;
	int m_aiPadSize[2];
	int m_aiNumTiles[2];
	int m_aiOffset[2];
	int m_iOverlap;
	float m_fOverlap;
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
     (  float* gfSpectrum,
        int* piCmpSize,
        CCTFTheory* pCTFTheory,
        float* pfResRange
     );
     float* GetSpectrum(bool bClean);
     int m_aiImgSize[2];
private:
     void mCreateFullSpectrum(float* gfSpectrum);
     void mEmbedCTF(void);
     float* m_pfSpectrum;
     CCTFTheory* m_pCTFTheory;
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
	void DoIt
	(  CCTFTheory* pCTFTheory,
	   float* pfResRange, // A
	   float* pfDfRange,  // A
	   float* gfRadialAvg,
	   int iSize
	);
	float m_fBestDf;
	float m_fMaxCC;
private:
	void mBrutalForceSearch(float fDfStep, float* pfDefocus, float* pfCC);
	void mCalcCTF(float fDefocus);
	float mCorrelate(void);
	CCTFTheory* m_pCTFTheory;
	float m_afResRange[2];
	float m_afDfRange[2];
     float* m_gfRadialAvg;
     int m_iSize;
     float* m_gfCTF1D;
};

class CFindCTF1D
{
public:
     CFindCTF1D(void);
     ~CFindCTF1D(void);
     void DoIt
     (  float* pfImage,
        int* piImgSize,
        int iTileSize,
        CCTFTheory* pCTFTheory
     );
     void DoIt
     (  float* pfImage,
        int* piImgSize,
        int iTileSize,
        CCTFTheory* pCTFTheory,
        float fDfGuess // A
     );
     void DoIt
     (  float* pfImage,
        int* piImgSize,
        int iTileSize,
        CCTFTheory* pCTFTheory,
        float* pfResRange,
        float* pfDfRange
     );
     void SaveSpectrum
     (  char* pcMrcFile
     );
     float m_fDefocus;
     float m_fCC;
protected:
     void mFindDefocus(float* gfRadiaAvg);
     void mGenSpectrum(float* pfImage);
     void mRemoveBackground(void);
     void mGenFullSpectrum(void);
     float* mCalcRadialAverage(void);
     int m_aiImgSize[2];
     int m_iTileSize;
     CCTFTheory* m_pCTFTheory;
     float m_afResRange[2];
     float m_afDfRange[2];
     int m_aiCmpSize[2];
     float* m_gfSpectrum;
     float* m_pfFullSpectrum; 
};

}
