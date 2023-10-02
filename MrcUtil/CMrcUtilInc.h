#pragma once
#include "../Util/CUtilInc.h"
#include <Util/Util_Thread.h>
#include <Mrcfile/CMrcFileInc.h>
#include <queue>

namespace MrcUtil
{
//-------------------------------------------------------------------
// CAlignParam stores the alignment parameters of all tilt images in
// an input tilt series.
// 1. Since tilt images may not be in order in the input MRC file,
//    CAlignParam maintains two indices: 1) Frame index orders
//    frames according to tilt angles; 2) Section index tracks the
//    order of tilt images in the input MRC file.
// 2. Dark images are removed after loading the MRC file. There are
//    no entries for the dark frames.
//-------------------------------------------------------------------
class CAlignParam
{
public:
	CAlignParam(void);
	~CAlignParam(void);
	void Clean(void);
	void Create(int iNumFrames);
	void SetSecIndex(int iFrame, int iSecIdx);
	void SetTilt(int iFrame, float fTilt);
	void SetTiltAxis(int iFrame, float fTiltAxis);
	void SetTiltAxisAll(float fTiltAxis);
	void SetShift(int iFrame, float* pfShift);
	void SetCenter(float fCentX, float fCentY);
	void SetTiltRange(float fEndAng1, float fEndAng2);
	//------------------------------------------------
	int GetSecIndex(int iFrame);
	float GetTilt(int iFrame);
	float GetTiltAxis(int iFrame);
	void GetShift(int iFrame, float* pfShift);
	float* GetShiftXs(void);  // do not free
	float* GetShiftYs(void);  // do not free
	int GetFrameIdxFromTilt(float fTilt);
	float* GetTilts(bool bCopy);
	float GetMinTilt(void);
	float GetMaxTilt(void);
	//---------------------
	void AddTiltOffset(float fTiltOffset);
	void AddShift(int iFrame, float* pfShift);
	void AddShift(float* pfShift);
	void MultiplyShift(float fFactX, float fFactY);
	//---------------------------------------------
	void RotateShift(int iFrame, float fAngle);
	static void RotShift(float* pfInShift,
	  float fRotAngle, float* pfOutShift);
	//------------------------------------
	void FitRotCenterX(void);
	void FitRotCenterZ(void);
	void GetRotationCenter(float* pfCenter); // x, y, z
	void SetRotationCenterZ(float fCentZ) { m_fZ0 = fCentZ; }
	void RemoveOffsetX(float fFact); // -1 remove, 1 restore
	void RemoveOffsetZ(float fFact); // -1 remove, 1 restore
	void CalcZInducedShift(int iFrame, float* pfShift);	
	//-------------------------------------------------
	void CalcRotCenter(void);
	//-----------------------
	void MakeRelative(int iRefFrame);
	void ResetShift(void);
	void SortByTilt(void);
	void SortBySecIndex(void);
	void RemoveFrame(int iFrame);
	//---------------------------
	CAlignParam* GetCopy(void);
	CAlignParam* GetCopy(int iStartFm, int iNumFms);
	CAlignParam* GetCopy(float fStartTilt, float fEndTilt);
	void Set(CAlignParam* pAlignParam);
	//---------------------------------
	void LogShift(char* pcLogFile);
	int m_iNumFrames;
private:
	void mSwap(int iFrame1, int iFrame2);
	int* m_piSecIndex;
	float* m_pfTilts;
	float* m_pfTiltAxis;
	float* m_pfShiftXs;
	float* m_pfShiftYs;
	float* m_pfDoses; // accumulated dose
	float m_afCenter[2];
	float m_afTiltRange[2];
	float m_fX0;
	float m_fY0;
	float m_fZ0;
};

class CLocalAlignParam
{
public:
	CLocalAlignParam(void);
	~CLocalAlignParam(void);
	void Clean(void);
	void Setup(int iNumTilts, int iNumPatches);
	void GetParam(int iTilt, float* gfAlnParam);
	void GetCoordXYs(int iTilt, float* pfCoordXs, float* pfCoordYs);
	void SetCoordXY(int iTilt, int iPatch, float fX, float fY);
	void SetShift(int iTilt, int iPatch, float fSx, float fSy);
	void SetBad(int iTilt, int iPatch, bool bBad);	
	float* m_pfCoordXs;
	float* m_pfCoordYs;
	float* m_pfShiftXs;
	float* m_pfShiftYs;
	float* m_pfGoodShifts;
	int m_iNumTilts;
	int m_iNumPatches;
	int m_iNumParams; // x,y,sx,sy,bad per tilt
};

class CPatchShifts
{
public:
	CPatchShifts(void);
	~CPatchShifts(void);
	void Clean(void);
	void Setup(int iNumPatches, int iNumTilts);
	void SetRawShift(int iPatch, CAlignParam* pPatAlnParam);
	void GetPatCenterXYs(float* pfCentXs, float* pfCentYs);
	void GetPatShifts(float* pfShiftXs, float* pfShiftYs);
	void RotPatCenterXYs(float fRot, float* pfCentXs, float* pfCentYs);
	//-----------------------------------------------------------------
	CAlignParam* GetAlignParam(int iPatch); // do not free
	void GetShift(int iPatch, int iTilt, float* pfShift);
	float GetTiltAxis(int iPatch, int iTilt);
	void GetRotCenter(int iPatch, float* pfRotCent); // x, y, z
	void SetRotCenterZ(int iPatch, float fCentZ);
	//-------------------------------------------

	int m_iNumPatches;
	int m_iNumTilts;
	bool* m_pbBadShifts;
private:
	MrcUtil::CAlignParam** m_ppPatAlnParams;
	int m_iZeroTilt;
};

class CTiltDoses
{
public:
	static CTiltDoses* GetInstance(void);
	static void DeleteInstance(void);
	~CTiltDoses(void);
	void Clean(void);
	void Setup(CAlignParam* pAlignParam);
	float* GetDoses(void); // do not delete
	float GetDose(int iImage);
	bool m_bDoseWeight;
	int m_iNumImgs;
private:
	float* m_pfDoses;
	CTiltDoses(void);
	static CTiltDoses* m_pInstance;
};

class CTomoStack
{
public:
	CTomoStack(void);
	~CTomoStack(void);
	void Clean(void);
	int GetPixels(void); // frame pixels
	int GetNumFrames(void);
	void Create(int* piStkSize, bool bAlloc);
	//---------------------------------------
	void SetFrame(int iFrame, float* pfFrame);
	float* GetFrame(int iFrame);  // do not free
	void GetFrame(int iFrame, float* pfFrame);
	//----------------------------------------
	void SetCenter(int iFrame, float* pfCent);
	void GetCenter(int iFrame, float* pfCent);
	//----------------------------------------
	CTomoStack* GetCopy(void);
	CTomoStack* GetSubStack
	( int* piStart,  // 3 elements
	  int* piSize    // 3 elements
	);
	void RemoveFrame(int iFrame);
	void GetAlignedFrameSize(float fTiltAxis, int* piAlnSize);
	//--------------------------------------------------------
	int m_aiStkSize[3];
	float** m_ppfFrames;
	float m_fPixSize;
private:
	void mCleanFrames(void);
	void mCleanCenters(void);
	float** m_ppfCenters;
};

class CRemoveDarkFrames : public Util_Thread
{
public:
	static void DoIt
	( CTomoStack* pTomoStack,
	  CAlignParam* pAlignParam,
	  float fThreshold,
	  int* piGpuIDs,
	  int iNumGpus
	);
	CRemoveDarkFrames(void);
	~CRemoveDarkFrames(void);
	void Run
	( Util::CNextItem* pNextItem,
	  int iGpuID,
	  float* pfMeans,
	  float* pfStds
	);
	void ThreadMain(void);
private:
	int m_iGpuID;
	Util::CNextItem* m_pNextItem;
	float* m_pfMeans;
	float* m_pfStds;
};

class CCalcStackStats : public Util_Thread
{
public:
	static void DoIt
	( CTomoStack* pTomostack, float* pfStats,
	  int* piGpuIDs, int iNumGpus
	);
	~CCalcStackStats(void);
	void Run
	( Util::CNextItem* pNextItem,
	  int iGpuID, float* pfFrmStats
	);
	void ThreadMain(void);
private:
	CCalcStackStats(void);
	int m_iGpuID;
	Util::CNextItem* m_pNextItem;
	float* m_pfFrmStats;	
};

class CDarkFrames
{
public:
	static CDarkFrames* GetInstance(void);
	static void DeleteInstance(void);
	~CDarkFrames(void);
	void Setup(int* piRawStkSize); // original tilts
	void Add(int iFrmIdx, int iSecIdx, float fTilt);
	void AddTiltOffset(float fTiltOffset);
	int GetFrmIdx(int iNthDark);
	int GetSecIdx(int iNthDark);
	float GetTilt(int iNthDark);
	bool IsDarkSection(int iSection); // index in input MRC
	bool IsDarkFrame(int iFrame);
	void GenImodExcludeList(char* pcLine, int iSize);
	int m_aiRawStkSize[3];
	int m_iNumDarks;
private:
	CDarkFrames(void);
	void mClean(void);
	int* m_piFrmIdxs; // frame index ordered by tilts
	int* m_piSecIdxs; // section index ordered in orginal MRC file
	float* m_pfTilts; // tilt angles of dark images
	static CDarkFrames* m_pInstance;
};

class CAcqSequence
{
public:
	static CAcqSequence* GetInstance(void);
	static void DeleteInstance(void);
	~CAcqSequence(void);
	void Clean(void);
	void ReadAngFile(char* pcAngFile);
	void AddTiltOffset(float fTiltOffset);
	void SortByAcquisition(void);
	int GetAcqIndexFromTilt(float fTilt);
	int GetAcqIndexFromSection(int iSecIdx);
	int GetAcqIndex(int iEntry);
	int GetSecIndex(int iEntry);
	float GetTiltAngle(int iEntry);
	void SaveCsv(char* pcCsvName, int iOutImod);
	bool hasSequence(void);
	int m_iNumSections;
private:
	CAcqSequence(void);
	void mSaveCsvRelion(char* pcCsvName);
	void mSaveCsvWarp(char* pcCsvName);
	int* m_piAcqIndices;
	int* m_piSecIndices;
	float* m_pfTiltAngles;
	static CAcqSequence* m_pInstance;
};

class CFindObjectCenter : public Util_Thread
{
public:
	static void DoIt
	( CTomoStack* pTomoStack,
	  CAlignParam* pAlignParam,
	  int* piGpuIDs,
	  int iNumGpus,
	  float* pfShift
	);
	CFindObjectCenter(void);
	~CFindObjectCenter(void);
	void Run
	( Util::CNextItem* m_pNextItem,
	  int GrpuID
	);
	void ThreadMain(void);
private:
	int m_iGpuID;
	Util::CNextItem* m_pNextItem;
};


class CSaveAlignFile
{
public:
	CSaveAlignFile(void);
	~CSaveAlignFile(void);
	void DoIt
	( char* pcInMrcFile,
	  char* pcOutMrcFile,
	  CAlignParam* pAlignParam,
	  CLocalAlignParam* pLocalParam
  	);
private:
	void mSaveHeader(void);
	void mSaveGlobal(void);
	void mSaveLocal(void);
	void mCloseFile(void);
	CAlignParam* m_pAlignParam;
	CLocalAlignParam* m_pLocalParam;
	void* m_pvFile;
	int m_iNumTilts;
	int m_iNumPatches;
};

class CLoadAlignFile
{
public:
	CLoadAlignFile(void);
	~CLoadAlignFile(void);
	bool DoIt(const char* pcAlnFile);
	CAlignParam* GetAlignParam(bool bClean);
	CLocalAlignParam* GetLocalParam(bool bClean);
private:
	bool mParseHeader(void);
	bool mParseRawSize(char* pcLine);
	bool mParseDarkFrame(char* pcLine);
	bool mParseNumPatches(char* pcLine);
	void mLoadGlobal(void);
	void mLoadLocal(void);
	void mClean(void);
	int m_iNumPatches;
	CAlignParam* m_pAlignParam;
	CLocalAlignParam* m_pLocalParam;
	std::queue<char*> m_aHeaderQueue;
	std::queue<char*> m_aDataQueue;
};

class CLoadAlignment
{
public:
	static CLoadAlignment* GetInstance(void);
	static void DeleteInstance(void);
	~CLoadAlignment(void);
	bool DoIt(void);
	CAlignParam* GetAlignParam(bool bClean);
	CLocalAlignParam* GetLocalParam(bool bClean);
	bool m_bFromAlnFile;
	bool m_bFromTiltRange;
	bool m_bFromAngFile;
	bool m_bFromHeader;
private:
	CLoadAlignment(void);
	void mClean(void);
	void mDoAlnFile(void);
	void mDoTiltRange(void);
	bool mDoAngFile(void);
	void mDoMrcFile(void);
	void mReadHeaderSections(void);
	CAlignParam* m_pAlignParam;
	CLocalAlignParam* m_pLocalParam;
	int m_iNumSections; // number of images in MRC file
	static CLoadAlignment* m_pInstance;
};

class CLoadStack
{
public:
	static CLoadStack* GetInstance(void);
	static void DeleteInstance(void);
	~CLoadStack(void);
	CTomoStack* DoIt(CAlignParam* pAlignParam);
	CTomoStack* GetStack(bool bClean);
	int m_aiStkSize[3];
private:
	CLoadStack(void);
	void mLoadShort(void);
	void mLoadUShort(void);
	void mLoadFloat(void);
	void mPrintStackInfo(int* piStkSize, int iMode);
	//----------------------------------------------
	CTomoStack* m_pTomoStack;
	CAlignParam* m_pAlignParam;
	Mrc::CLoadMrc m_aLoadMrc;
	static CLoadStack* m_pInstance;
};	//CLoadStacks


class CSaveStack
{
public:
	CSaveStack(void);
	~CSaveStack(void);
	bool OpenFile(char* pcMrcFile);
	void DoIt
	( CTomoStack* pTomoStack,
	  CAlignParam* pAlignParam,
	  float fPixelSize,
	  float* pfStats,
	  bool bVolume
	);
private:
	void mDrawTiltAxis(float* pfImg, int* piSize, float fTiltAxis);
	Mrc::CSaveMrc m_aSaveMrc;
	char m_acMrcFile[256];
};

class GExtractPatch
{
public:
	GExtractPatch(void);
	~GExtractPatch(void);
	void SetStack
	( CTomoStack* pTomoStack,
	  CAlignParam* pAlignParam
	);
	CTomoStack* DoIt
	( int* piStart, // 3 elements
	  int* piSize   // 3 elements
	);
private:
	void mExtract(void);
	void mCalcCenter
	( float* pfCent0,
	  float fTilt,
	  float fTiltAxis,
	  float* pfCent
	);
	void mExtractProj
	( int iProj,
	  float* pfCent,
	  float* pfPatch
	);
	int m_aiStart[3];
	int m_aiSize[3];
	CTomoStack* m_pTomoStack;
	CAlignParam* m_pAlignParam;
	CTomoStack* m_pTomoPatch;
};

class CGenCentralSlices
{
public:
	CGenCentralSlices(void);
	~CGenCentralSlices(void);
	void Clean(void);
	void DoIt(CTomoStack* pVolStack); // must be xzy order
	void GetSizeXY(int* piSize);
	void GetSizeYZ(int* piSize);
	void GetSizeXZ(int* piSize);
	float* m_pfSliceXY;
	float* m_pfSliceYZ;
	float* m_pfSliceXZ;
private:
	void mIntegrateX(CTomoStack* pVolStack);
	void mIntegrateY(CTomoStack* pVolStack);
	void mIntegrateZ(CTomoStack* pVolStack);
	int m_iSizeX;
	int m_iSizeY;
	int m_iSizeZ;
};

class CCropVolume
{
public:
	CCropVolume(void);
	~CCropVolume(void);
	void Clean(void);
	CTomoStack* DoIt(CTomoStack* pInVol, float fOutBin,
	   CAlignParam* pFullParam,
	   CLocalAlignParam* pLocalParam,
	   int* piOutSizeXY);
private:
	void mCalcOutCenter(void);
	void mCreateOutVol(void);
	void mCalcOutVol(void);	
	CTomoStack* m_pInVol; // must be xzy x is fastest y slowest
	CAlignParam* m_pFullParam;
	CLocalAlignParam* m_pLocalParam;
	float m_fOutBin;
	int m_aiOutSize[2];
	int m_aiOutCent[2];
	CTomoStack* m_pOutVol;
};
}
