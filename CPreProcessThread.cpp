#include "CPreProcessThread.h"
#include "CInput.h"
#include "CFFTBuffer.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include "FindTilts/CFindTiltsInc.h"
#include "StreAlign/CStreAlignInc.h"
#include "ProjAlign/CProjAlignInc.h"
#include "PatchAlign/CPatchAlignInc.h"
#include "CommonLine/CCommonLineInc.h"
#include "Massnorm/CMassNormInc.h"
#include "TiltOffset/CTiltOffsetInc.h"
#include "Recon/CReconInc.h"
#include "DoseWeight/CDoseWeightInc.h"
#include "FindCtf/CFindCtfInc.h"
#include "ImodUtil/CImodUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Util/Util_Time.h>
#include <iostream>


namespace SA = StreAlign;
namespace PA = ProjAlign;

CPreProcessThread::CPreProcessThread(void)
{
	m_pTomoStack = 0L;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
	m_pCorrTomoStack = 0L;
}

CPreProcessThread::~CPreProcessThread(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
}

bool CPreProcessThread::DoIt(MrcUtil::CTomoStack* pTomoStack)
{	
	bool bExit = this->WaitForExit(10000000.0f);
	if(!bExit) return false;
	//----------------------
	m_pTomoStack = pTomoStack;
	this->Start();
	return true;
}

void CPreProcessThread::ThreadMain(void)
{
	printf("\nProcess thread has been started.\n\n");
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[0]);
	//-----------------------------------
	MrcUtil::CLoadAlignment* pLoadAlign = 
	   MrcUtil::CLoadAlignment::GetInstance();
	m_pAlignParam = pLoadAlign->GetAlignParam(true);
	m_pLocalParam = pLoadAlign->GetLocalParam(true);
	//----------------------------------------------
	PA::CRemoveSpikes::DoIt(m_pTomoStack, pInput->m_piGpuIDs,
	   pInput->m_iNumGpus);
	mSetPositivity();	
	//---------------
	mFindCtf();
}

void CPreProcessThread::mFindCtf(void)
{
	FindCtf::CFindCtfMain aFindCtfMain;
	if(!aFindCtfMain.CheckInput()) return;
	//------------------------------------
	aFindCtfMain.DoIt(m_pTomoStack, m_pAlignParam);
}

void CPreProcessThread::mSetPositivity(void)
{
	MassNorm::GPositivity aGPositivity;
	aGPositivity.DoIt(m_pTomoStack);
}

