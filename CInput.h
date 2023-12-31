#pragma once
#include <stdio.h>

class CInput
{
public:
	static CInput* GetInstance(void);
	static void DeleteInstance(void);
	~CInput(void);
	void ShowTags(void);
	void Parse(int argc, char* argv[]);
	char* GetLogFile(char* pcSuffix, int* piSerial = 0L);
	char* GetTmpFile(char* pcSuffix, int* piSerial = 0L);
	float GetOutPixSize(void);
	char m_acInMrcFile[256];
	char m_acOutMrcFile[256];
	char m_acAlnFile[256];
	char m_acAngFile[256];
	char m_acRoiFile[256];
	char m_acTmpFile[256];
	char m_acLogFile[256];
	float m_afTiltRange[2];
	float m_afTiltAxis[2];
	int m_iAlignZ;
	int m_iVolZ;
	float m_fOutBin;
	int* m_piGpuIDs;
	int m_iNumGpus;
	float m_afTiltCor[2];
	float m_afReconRange[2];
	float m_fPixelSize;
	float m_fKv;
	float m_fImgDose;
	float m_fCs;
	float m_fAmpContrast;
	float m_afExtPhase[2];
	int m_iFlipVol;
	int m_iFlipInt;
	int m_aiSartParam[2];
	int m_iWbp;
	int m_aiNumPatches[2];
	float m_afTiltScheme[3];
	int m_aiCropVol[2];
	int m_iOutXF;
	int m_iAlign;
	int m_iOutImod;
	float m_fDarkTol;
	float m_afBFactor[2];
	int m_iIntpCor;
	//-------------
        char m_acInMrcTag[32];
        char m_acOutMrcTag[32];
	char m_acAlnFileTag[32];
	char m_acAngFileTag[32];
	char m_acRoiFileTag[32];
        char m_acTmpFileTag[32];
	char m_acLogFileTag[32];
	char m_acTiltRangeTag[32];
	char m_acTiltAxisTag[32];
	char m_acAlignZTag[32];
	char m_acVolZTag[32];
	char m_acOutBinTag[32];
	char m_acTiltCorTag[32];
        char m_acGpuIDTag[32];
	char m_acReconRangeTag[32];
	char m_acPixelSizeTag[32];
	char m_acKvTag[32];
	char m_acCsTag[32];
	char m_acAmpContrastTag[32];
	char m_acExtPhaseTag[32];
	char m_acImgDoseTag[32];
	char m_acFlipVolTag[32];
	char m_acFlipIntTag[32];
	char m_acSartTag[32];
	char m_acWbpTag[32];
	char m_acPatchTag[32];
	char m_acTiltSchemeTag[32];
	char m_acOutXFTag[32];
	char m_acAlignTag[32];
	char m_acCropVolTag[32];
	char m_acOutImodTag[32];
	char m_acDarkTolTag[32];
	char m_acBFactorTag[32];
	char m_acIntpCorTag[32];
private:
        CInput(void);
        void mPrint(void);
	char* mGenFileName(char* pcPrefix, char* pcSuffix, int* piSerial);
        int m_argc;
        char** m_argv;
        static CInput* m_pInstance;
};
