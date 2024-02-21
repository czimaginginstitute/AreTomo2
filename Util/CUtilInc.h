#pragma once
#include <Mrcfile/CMrcFileInc.h>
#include <cufft.h>
#include <pthread.h>

namespace Util
{

size_t GetFloatBytes(int* piSize);
size_t GetCmpBytes(int* piSize);
size_t GetUCharBytes(int* piSize);
float* GGetFBuf(int* piSize, bool bZero);
cufftComplex* GGetCmpBuf(int* piSize, bool bZero);
unsigned char* GGetUCharBuf(int* piSize, bool bZero);
void PrintGpuMemoryUsage(const char* pcInfo);
void CheckCudaError(const char* pcLocation);
//------------------------------------------

class CParseArgs
{
public:
        CParseArgs(void);
        ~CParseArgs(void);
        void Set(int argc, char* argv[]);
        bool FindVals(const char* pcTag, int aiRange[2]);
        void GetVals(int aiRange[2], float* pfVals);
        void GetVals(int aiRange[2], int* piVal);
        void GetVal(int iArg, char* pcVal);
        void GetVals(int aiRange[2], char** ppcVals);
private:
        char** m_argv;
        int m_argc;
};

class CNextItem
{
public:
	CNextItem(void);
	~CNextItem(void);
	void Create(int iNumItems);
	void Reset(void);
        int GetNext(void);
private:
	int m_iNumItems;
	int m_iNextItem;
	pthread_mutex_t m_aMutex;
};

class CSplitItems
{
public:
	CSplitItems(void);
	~CSplitItems(void);
	void Clean(void);
	void Create(int iNumItems, int iNumSplits);
	int GetStart(int iSplit);
	int GetSize(int iSplit);
	int m_iNumSplits;
private:
	int m_iNumItems;
	int* m_piStarts;
	int* m_piSizes;
};

class CPad2D
{
public:
	CPad2D(void);
	~CPad2D(void);
	static void GetPadSize(int* piImgSize, int* piPadSize);
	static void GetImgSize(int* piPadSize, int* piImgSize);
	static float* GGetPadBuf(int* piImgSize, bool bZero);
	static float* CGetPadBuf(int* piImgSize, bool bZero);
	static float* GGetImgBuf(int* piPadSize, bool bZero);
	static float* CGetImgBuf(int* piPadSize, bool bZero);

	float* GPad(float* pfImg, int* piSize);
	float* CPad(float* pfImg, int* piSize);
	void Pad(float* pfImg, int* piImgSize, float* pfPad);
	float* GUnpad(float* pfPad, int* piSize);
	float* CUnpad(float* pfPad, int* piSize);
	void Unpad(float* pfPad, int* piPadSize, float* pfImg);
private:
	static float* mGGetBuf(int* piSize, bool bZero);
	static float* mCGetBuf(int* piSize, bool bZero);
};      //CPad2D

class CFileName
{
public:
	CFileName(const char* pcFileName);
	CFileName(void);
	~CFileName(void);
	void Setup(const char* pcFileName);
	void GetFolder(char* pcFolder);
	void GetName(char* pcName); // file name no path & no extension
	void GetExt(char* pcExt);
private:
	char m_acFolder[256];
	char m_acFileName[128];
	char m_acFileExt[32];
};

class GAddImages
{
public:
	GAddImages(void);
	~GAddImages(void);
	void DoIt(float* gfImage1, float fFactor1,
	   float* gfImage2, float fFactor2,
	   float* gfSum, int* piImgSize);
};

class GCalcMoment2D
{
public:
	GCalcMoment2D(void);
	~GCalcMoment2D(void);
	void Clean(void);
	void SetSize(int* piImgSize, bool bPadded);
	float DoIt(float* gfImg, int iExponent, bool bSync,
	   cudaStream_t stream = 0);
	float GetResult(void);
	void Test(float* gfImg, float fExp);
private:
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
	float* m_gfBuf;
	int m_iPadX;
	int m_aiImgSize[2];
};

class GFindMinMax2D
{
public:
	GFindMinMax2D(void);
	~GFindMinMax2D(void);
	void Clean(void);
	void SetSize(int* piImgSize, bool bPadded);
	float DoMin(float* gfImg, bool bSync, cudaStream_t stream = 0);
	float DoMax(float* gfImg, bool bSync, cudaStream_t stream = 0);
	float GetResult(void);
	void Test(float* gfImg);
private:
	int m_aiImgSize[2];
	int m_iPadX;
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
	float* m_gfBuf;
};

class GThreshold2D
{
public:
	GThreshold2D(void);
	~GThreshold2D(void);
	void DoIt(float* gfImg, float fMin, float fMax,
	   int* piImgSize, bool bPadded);
};

class GFFT1D
{
public:
	GFFT1D(void);
	~GFFT1D(void);
	void DestroyPlan(void);
	void CreatePlan
	( int iFFTSize,
	  int iNumLines,
	  bool bForward
	);
	void Forward(float* gfPadLines,bool bNorm);
	void Inverse(cufftComplex* gCmpLines);
private:
        int m_iFFTSize;
        int m_iNumLines;
        cufftType m_cufftType;
        cufftHandle m_cufftPlan;
};

class GFFT2D
{
public:
	GFFT2D(void);
	~GFFT2D(void);
	void SetStream(cudaStream_t stream);
	void DestroyPlan(void);
	void CreatePlan(int* piFFTSize, bool bForward);
	void Forward(float* gfPadImg, bool bNorm);
	void Forward(float* gfImg, cufftComplex* gCmp, bool bNorm);
	void Inverse(cufftComplex* gCmp);
	void Inverse(cufftComplex* gCmp, float* gfImg);
	void RemoveAmp(cufftComplex* gCmp, int* piCmpSize);
private:
	void mNormalize(cufftComplex* gCmpImg);
	void mCheckError(cufftResult error, const char* pcFunc);
	int m_aiFFTSize[2];
	cufftType m_cufftType;
	cufftHandle m_cufftPlan;
	cudaStream_t m_aStream;
};

class GFFTUtil2D
{
public:
	GFFTUtil2D(void);
	~GFFTUtil2D(void);
	void Multiply
	( cufftComplex* gComp,
	  int* piCmpSize,
	  float fFactor
	);
	void GetAmp
	( cufftComplex* gComp,
	  int* piCmpSize,
	  float* pfAmpRes,
	  bool bGpuRes
	);
	void Shift
	( cufftComplex* gComp, 
	  int* piCmpSize,
	  float* pfShift
	);
};

class GRoundEdge
{
public:
	GRoundEdge(void);
	~GRoundEdge(void);
	void SetMask(float* pfCent, float* pfSize);
	void DoIt
	( float* gfImg, int* piSize, bool bPadded, 
	  float fPower, cudaStream_t stream = 0
	);
private:
	float m_afMaskCent[2];
	float m_afMaskSize[2];
};	//GRoundEdge

class GRoundEdge1D
{
public:
	GRoundEdge1D(void);
	~GRoundEdge1D(void);
	void DoIt(float* gfData, int iSize);
	void DoPad(float* gfPadData, int iPadSize);
};

class GStretch
{
public:
	GStretch(void);
	~GStretch(void);
	void Clean(void);
	void CalcMatrix
	( float fStretch, // compress when < 1
	  float fTiltAxis
	);
	void DoIt
	( float* gfInImg,   // input image
	  int* piSize,
	  bool bPadded,
	  float fStretch, // if less than 1, then compress 
	  float fTiltAxis,
	  float* gfOutImg,   // output image
	  bool bRandomFill = false,
	  cudaStream_t stream = 0
	);
	void Unstretch
	( float* pfInShift,
	  float* pfOutShift
	);
private:
	float m_afMatrix[3];
	float* m_gfMatrix;
};

class GMutualMask2D
{
public:
	GMutualMask2D(void);
	~GMutualMask2D(void);
	void DoIt(float* gfImg1, float* gfImg2,
	   int* piSize, bool bPadded,
	   cudaStream_t stream);
};

class CPeak2D
{
public:
	CPeak2D(void);
	~CPeak2D(void);
	void GetShift(float fXcfBin, float* pfShift);
	void GetShift(float* pfXcfBin, float* pfShift);
	void DoIt
	( float* pfImg, int* piImgSize, bool bPadded,
	  int* piSeaSize = 0L
	);
	float m_afShift[2];
	float m_fPeakInt;
private:
	void mSearchIntPeak(void);
	void mSearchFloatPeak(void);
	float* m_pfImg;
	int m_aiImgSize[2];
	int m_iPadX;
	int m_aiSeaSize[2];
	int m_aiPeak[2];
	float m_afPeak[2];
};	

class GXcf2D
{
public:
	GXcf2D(void);
	~GXcf2D(void);
	void Clean(void);
	void Setup(int* piCmpSize);
	void DoIt
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  float fBFactor	  
	);
	float SearchPeak(void);
	float* GetXcfImg(bool bClean);
	void GetShift
	( float* pfShift, 
	  float fXcfBin
	);
	int m_aiXcfSize[2];
	float* m_pfXcfImg;
	float m_fPeak;
private:
	float m_afShift[2];
	float m_fBFactor;
	GFFT2D m_fft2D;
};

class GRotate2D
{
public:
	GRotate2D(void);
	~GRotate2D(void);
	void Clear(void);
	void SetFillValue(float fFillVal);
	void SetImage(float* gfInImg, int* piSize, bool bPadded);
	void DoIt(float fAngle, float* gfOutImg, bool bPadded);
private:
	void mCalcTrig(float fAngle);
	int m_aiSize[2];
	float m_fFillVal;
	float* m_gfInImg;
	int m_aiImgSize[2];
};


class GShiftRotate2D
{
public:
	GShiftRotate2D(void);
	~GShiftRotate2D(void);
	void SetSizes
	( int* piInSize,
	  bool bInPadded,
	  int* piOutSize,
	  bool bOutPadded
	);
	void DoIt
	( float* gfInImg,
	  float* pfShift,
	  float fRotAngle,
	  float* gfOutImg,
	  bool bRandomFill,
	  cudaStream_t stream = 0
	);
	int m_aiOutSize[2];
private:
	int m_iInImgX;
	int m_iOutImgX;
	int m_iOutImgY;
};


//-------------------------------------------------------------------
// 1. Subtract mean of the input array pfData from each element.
// 2. Only the positive elements are counted in the mean calculation.
// 3. During the subtraction, the negative elements are set zero. 
//-------------------------------------------------------------------
class GNormalize2D
{
public:
	GNormalize2D(void);
	~GNormalize2D(void);
	void DoIt
	( float* gfImg, int* piSize, bool bPadded,
	  float fMean, float fStd, cudaStream_t stream = 0
	);
};

//-------------------------------------------------------------------
//  1. Detect spike pixels in a 5x5 area. Spike pixels are those
//     that are outside m_iNumSigmas around local mean.
//  2. Spiky pixels are replaced by the local mean.
//  3. By default m_iNumSigmas is 3.
//-------------------------------------------------------------------
class GRemoveSpikes2D
{
public:
	GRemoveSpikes2D(void);
	~GRemoveSpikes2D(void);
	void DoIt
	( float* gfInImg, int* piImgSize, bool bPadded,
	  int iWinSize, float* gfOutImg,
	  cudaStream_t stream = 0
	);
};

class GBinImage2D
{
public:
	GBinImage2D(void);
	~GBinImage2D(void);
	static void GetBinSize
	( int* piInSize, bool bInPadded, int* piBinning,
	  int* piOutSize, bool bOutPadded
	);
	static void GetBinSize
	( int* piInSize, bool bInPadded, int iBinning,
	  int* piOutSize, bool bOutPadded
	);
	void SetupBinnings
	( int* piInSize,  bool bInPadded,
	  int* piBinning, bool bOutPadded
	);
	void SetupBinning
	( int* piInSize, bool bInPadded,
	  int iBinning, bool bOutPadded
	);
	void SetupSizes
	( int* piInSize, bool bInPadded,
	  int* piOutSize, bool bOutPadded
	);
	void DoIt
	( float* gfInImg,  // input image
	  float* gfOutImg,
	  cudaStream_t stream = 0
	);
	int m_aiOutSize[2];
	int m_iOutImgX;
	int m_aiBinning[2];
};

class CStretchCC2D
{
public:
	CStretchCC2D(void);
	~CStretchCC2D(void);
	void Clean(void);
	void Setup
	( int* piImgSize, 
	  int iBinning,
	  float fBFactor
	);
	float DoIt
	( float* pfImg1, 
	  float* pfImg2, 
	  float* pfTilts, 
	  float fTiltAxis
	);	
	float m_fCCSum;
	float m_fStdSum;
	float m_fCC;
private:
	void mBinImages(float* pfImg1, float* pfImg2);
	void mStretch(float* pfTilts, float fTiltAxis);
	void mRemoveMean(void);
	void mRoundEdge(void);
	void mForwardFFT(void);
	GBinImage2D m_binImg2D;
	GStretch m_stretch;
	GFFT2D m_fft2D;
	cufftComplex* m_gCmp1;
	cufftComplex* m_gCmp2;
	int m_iBinning;
	int m_aiCCPadSize[2];
	int m_aiImgSize[2];
	float m_fBFactor;
};	

class GCC1D
{
public:
	GCC1D(void);
	~GCC1D(void);
	void SetBFactor(float fBFactor);
	float DoIt
	( cufftComplex* gCmp1, 
	  cufftComplex* gCmp2, 
	  int iCmpSize
	);
	float m_fCCSum;
	float m_fStdSum;
	float m_fCC;
private:
	int mCalcWarps(int iSize, int iWarpSize);
	void mTestOnCPU
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  int iCmpSize
	);
	float m_fBFactor;
};

class GRealCC2D
{
public:
	GRealCC2D(void);
	~GRealCC2D(void);
	void SetMissingVal(double dVal);
	float DoIt
	( float* gfImg1,
	  float* gfImg2,
	  float* gfBuf,
	  int* piSize,
	  bool bPad,
	  cudaStream_t stream = 0
	);
private:
	float m_fMissingVal;
};	

class GCC2D
{
public:
	GCC2D(void);
	~GCC2D(void);
	void SetBFactor(float fBFactor);
	float DoIt
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  int* piCmpSize
	);
	float m_fCC;
	float m_fCCSum;
	float m_fStdSum;
private:
	int mDo1D
	( float* gfCC,
	  float* gfStd,
	  float* gfCCRes,
	  float* gfStdRes,
	  int iSize
	);
	int mCalcWarps(int iSize, int iWarpSize);
	void mTest(cufftComplex* gCmp1, cufftComplex* gCmp2, int* piCmpSize);
	float m_fBFactor;
};

//-------------------------------------------------------------------
// 1. Decompose a 2D cartesian vector along tangential and radial
//    direction.
// 2. pfPos: position vector. Note it may not be unit vector.
// 3. pfInVector: the cartesian vector.
// 4. pfOutVector: the first component is the tangential and the
//    second is the radial.
// 5. Note: positive r-component is in the same direction of pfPos.
//    Positive t-component is that r-component x t-component is
//    along z-axis.
//-------------------------------------------------------------------
class CTRDecompose2D
{
public:
	CTRDecompose2D(void);
	~CTRDecompose2D(void);
	void DoIt
	(  float* pfPos,
	   float* pfInVector,
	   float* pfOutVector
	);
};
	
class GFourierCrop2D
{
public:
	GFourierCrop2D(void);
	~GFourierCrop2D(void);
	static void GetImgSize
	( int* piImgSizeIn,
          float fBin,
          int* piImgSizeOut
	);
	static void GetPadSize
	( int* piPadSizeIn,
	  float fBin,
	  int* piPadSizeOut
	);
	static void GetCmpSize
	( int* piCmpSize,
	  float fBin,
	  int* piCmpSizeOut
	);
        static void CalcBinning
        ( int* piSizeIn,
          int* piSizeOut,
	  bool bImgSize,   // true: image size, false: cmp size 
          float* pfBinning
        );
        void DoIt
        ( cufftComplex* gCmpIn,
          int* piSizeIn,
          bool bNormalized, // Is input normalized FT? 
          cufftComplex* gCmpOut,
          int* piSizeOut
        );
};

class GPositivity2D
{
public:
	GPositivity2D(void);
	~GPositivity2D(void);
	void DoIt(float* gfImg, int* piImgSize, cudaStream_t = 0);
	void AddVal(float* gfImg, int* piImgSize, float fVal,
	   cudaStream_t stream = 0);
};

class CRemoveSpikes1D
{
public:
	CRemoveSpikes1D(void);
	~CRemoveSpikes1D(void);
	void SetWinSize(int iWinSize);
	void DoIt(float* pfData, int iSize);
	void DoIt(float* pfDataX, float* pfDataY, int iSize);
private:
	bool mQuadraticFit(int iStart);
	void mCalcTerms(int x, float fW);
	float* m_pfDataX;
	float* m_pfDataY;
	float* m_pfOldY;
	float* m_pfWeight;
	int m_iSize;
	int m_iWinSize;
	int m_iDim;
	float* m_pfTerms;
	float* m_pfFit;
	float* m_pfMatrix;
};

class CSaveTempMrc
{
public:
        CSaveTempMrc(void);
        ~CSaveTempMrc(void);
        void SetFile(const char* pcMajor, const char* pcMinor);
	void GDoIt(cufftComplex* gCmp, int* piCmpSize);
	void GDoIt(float* gfImg, int* piSize);
	void GDoIt(unsigned char* gucImg, int* piSize);
	void DoIt(void* pvImg, int iMode, int* piSize);
private:
	char m_acMrcFile[256];
};	//CSaveTempMrc


class CStrNode
{
public:
	CStrNode(void);
	~CStrNode(void);
	char* m_pcString;
	CStrNode* m_pNextNode;
};

class CStrLinkedList
{
public:
	CStrLinkedList(void);
	~CStrLinkedList(void);
	void Add(char* pcString);
	char* GetString(int iNode);
	int m_iNumNodes;
private:
	void mClean(void);
	CStrNode* m_pHeadNode;
	CStrNode* m_pEndNode;
};

class CReadDataFile
{
public:
	CReadDataFile(void);
	~CReadDataFile(void);
	float GetData(int iRow, int iCol);
	void GetRow(int iRow, float* pfRow);
	bool DoIt(char* pcFileName, int iNumCols);
	int m_iNumRows;
private:
	void mClean(void);
	float* m_pfData;
	int m_iNumCols;
};

class GCorrLinearInterp
{
public:
	GCorrLinearInterp(void);
	~GCorrLinearInterp(void);
	void DoIt
	( cufftComplex* gCmpFrm, int* piCmpSize,
	  cudaStream_t stream = 0
	);
};
	
}
