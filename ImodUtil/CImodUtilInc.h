#pragma once
#include "../MrcUtil/CMrcUtilInc.h"

namespace ImodUtil
{
class CSaveXF
{
public:
	CSaveXF(void);
	~CSaveXF(void);
	void DoIt
	( MrcUtil::CAlignParam* pGlobalParam,
	  const char* pcFileName
	);
private:
	void mSaveForRelion(void);
	void mSaveForWarp(void);
	void mSaveForAligned(void);
	void* m_pvFile;
	MrcUtil::CAlignParam* m_pGlobalParam;
};

class CSaveTilts
{
public:
	CSaveTilts(void);
	~CSaveTilts(void);
	void DoIt
	( MrcUtil::CAlignParam* pAlignParam,
	  const char* pcFileName
	);
private:
	void mSaveForRelion(void);
	void mSaveForWarp(void);
	void mSaveForAligned(void);
	void* m_pvFile;
	MrcUtil::CAlignParam* m_pGlobalParam;
};

class CSaveCsv
{
public:
	CSaveCsv(void);
	~CSaveCsv(void);
	void DoIt
	( MrcUtil::CAlignParam* pAlignParam,
	  const char* pcFileName
	);
private:
	void mSaveForRelion(void);
        void mSaveForWarp(void);
        void mSaveForAligned(void);
        void* m_pvFile;
        MrcUtil::CAlignParam* m_pGlobalParam;
};

class CSaveXtilts
{
public:
	CSaveXtilts(void);
	~CSaveXtilts(void);
	void DoIt
	( MrcUtil::CAlignParam* pAlignParam,
	  const char* pcFileName
	);
};

class CImodUtil
{
public:
	static CImodUtil* GetInstance(void);
	static void DeleteInstance(void);
	~CImodUtil(void);
	void CreateFolder(void);
	void SaveTiltSeries
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam,
	  float fPixelSize
	);
	void SaveVolume
	( MrcUtil::CTomoStack* pVolStack,
	  float fPixelSize,
	  float* pfStats
	);
private:
	CImodUtil(void);
	void mSaveTiltSeries(void);
	void mSaveNewstComFile(void);
	void mSaveTiltComFile(void);
	void mSaveCtfFile(void);
	void mCreateFileName(const char* pcInFileName, char* pcOutFileName);
	char m_acOutFolder[256];
	char m_acInMrcFile[128];
	char m_acAliFile[128];
	char m_acTltFile[128];
	char m_acCsvFile[128];
	char m_acXfFile[128];
	char m_acXtiltFile[128];
	char m_acRecFile[128];
	char m_acCtfFile[128];
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CTomoStack* m_pVolStack;
	MrcUtil::CAlignParam* m_pGlobalParam;
	float m_fTiltAxis;
	float m_fPixelSize;
	static CImodUtil* m_pInstance;
};

}
