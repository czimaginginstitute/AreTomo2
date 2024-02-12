#pragma once
#include "MrcUtil/CMrcUtilInc.h"
#include "Correct/CCorrectInc.h"
#include <Util/Util_Thread.h>
#include <stdio.h>
#include <cufft.h>

class CPreProcessThread : public Util_Thread
{
public:
	CPreProcessThread(void);
	~CPreProcessThread(void);
	bool DoIt(MrcUtil::CTomoStack* pTomoStack);
	void ThreadMain(void);
private:
	void mLoadAlignment(void);
	void mFindCtf(void);
	void mSetPositivity(void);
	//--------------------
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
	MrcUtil::CLocalAlignParam* m_pLocalParam;
	Correct::CCorrTomoStack* m_pCorrTomoStack;
	bool m_bLoadAlnFile;
	float m_fRotScore;
	float m_fTiltOffset;
};

