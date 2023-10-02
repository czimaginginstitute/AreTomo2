#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Recon/CReconInc.h"
#include <Util/Util_Thread.h>
#include <stdio.h>
#include <cufft.h>

namespace TomoRalign
{

class CInput
{
public:
        static CInput* GetInstance(void);
        static void DeleteInstance(void);
        ~CInput(void);
	void ShowTags(void);
        void Parse(int argc, char* argv[]);
        char m_acInMrcFile[256];
        char m_acOutMrcFile[256];
        int m_aiVolSize[3];
        char m_acTmpFile[256];
	char m_acLogFile[256];
        int* m_piGpuIds;
	int m_iNumGpus;
	//-------------
        char m_acInMrcTag[32];
        char m_acOutMrcTag[32];
        char m_acVolSizeTag[32];
        char m_acTmpFileTag[32];
	char m_acLogFileTag[32];
        char m_acGpuIDTag[32];
private:
        CInput(void);
        void mPrint(void);
        int m_argc;
        char** m_argv;
        static CInput* m_pInstance;
};

class CProcessThread : public Util_Thread
{
public:
	CProcessThread(void);
	~CProcessThread(void);
	bool DoIt
	(  MrcUtil::CTomoStack* pTomoStack
	);
	void ThreadMain(void);
private:
	void mBin2x(void);
	void mDoIt(void);
	void mSaveTomoMrc(void);
	MrcUtil::CTomoStack* m_pTomoStack;
	Recon::CCalcProjs* m_pCalcProjs;
};

class CMain
{
public:
	CMain(void);
	~CMain(void);
	void DoIt(void);
private:
};	//CMain

}
