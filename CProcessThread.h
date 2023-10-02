#pragma once
#include "MrcUtil/CMrcUtilInc.h"
#include "Correct/CCorrectInc.h"
#include <Util/Util_Thread.h>
#include <stdio.h>
#include <cufft.h>

class CProcessThread : public Util_Thread
{
public:
	CProcessThread(void);
	~CProcessThread(void);
	bool DoIt(MrcUtil::CTomoStack* pTomoStack);
	void ThreadMain(void);
private:
	void mLoadAlignment(void);
	void mFindCtf(void);
	void mAlign(void);
	void mCoarseAlign(void);
	void mFindTilts(void);
	void mStretchAlign(void);
	void mProjAlign(void);
	void mRotAlign(float fAngRange, int iNumSteps);
	void mRotAlign(void);
	void mFindTiltOffset(void);
	void mPatchAlign(void);
	void mDoseWeight(void);
	void mSetPositivity(void);
	void mCorrectStack(void);
	void mCorrectForImod(void);
	void mRecon(void);
	void mCropVol(void);
	void mSartRecon(int iVolZ);
	void mWbpRecon(int iVolZ);
	void mFlipInt(void);
	void mFlipVol(void);
	void mSaveCentralSlices(void);
	void mSaveAlignment(void);
	void mSaveStack(void);
	//--------------------
	MrcUtil::CTomoStack* m_pTomoStack;
	MrcUtil::CAlignParam* m_pAlignParam;
	MrcUtil::CLocalAlignParam* m_pLocalParam;
	Correct::CCorrTomoStack* m_pCorrTomoStack;
	bool m_bLoadAlnFile;
	float m_fRotScore;
	float m_fTiltOffset;
};

