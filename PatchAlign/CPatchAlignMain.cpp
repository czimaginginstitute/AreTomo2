#include "CPatchAlignInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../ProjAlign/CProjAlignInc.h"
#include "../Massnorm/CMassNormInc.h"
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

using namespace PatchAlign;
static float s_fD2R = 0.0174533f;
static MrcUtil::CTomoStack* s_pTomoStack = 0L;
static MrcUtil::CAlignParam* s_pFullParam = 0L;
static MrcUtil::CLocalAlignParam* s_pLocalParam = 0L;
static MrcUtil::CPatchShifts* s_pPatchShifts = 0L;
static Util::CNextItem* s_pNextItem = 0L;
static float s_fTiltOffset = 0.0f;

CPatchAlignMain::CPatchAlignMain(void)
{
}

CPatchAlignMain::~CPatchAlignMain(void)
{
}

MrcUtil::CLocalAlignParam* CPatchAlignMain::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	float fTiltOffset
)
{	s_pTomoStack = pTomoStack;
	s_pFullParam = pAlignParam;
	s_fTiltOffset = fTiltOffset;
	//--------------------------
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[0]);
	//-----------------------------------
	CPatchTargets* pPatchTargets = CPatchTargets::GetInstance();
	s_pLocalParam = new MrcUtil::CLocalAlignParam;
	s_pLocalParam->Setup(s_pTomoStack->m_aiStkSize[2],
	   pPatchTargets->m_iNumTgts);
	//----------------------------
	if(s_pPatchShifts != 0L) delete s_pPatchShifts;
	s_pPatchShifts = new MrcUtil::CPatchShifts;
	s_pPatchShifts->Setup(pPatchTargets->m_iNumTgts, 
	   s_pTomoStack->m_aiStkSize[2]);
	//-------------------------------
	s_pNextItem = new Util::CNextItem;
	s_pNextItem->Create(pPatchTargets->m_iNumTgts);
	//---------------------------------------------
	CPatchAlignMain* pThreads = new CPatchAlignMain[pInput->m_iNumGpus];
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	pThreads[i].Run(i);
	}
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	delete s_pNextItem; s_pNextItem = 0L;
	//-----------------------------------
	CFitPatchShifts aFitPatchShifts;
	aFitPatchShifts.Setup(s_pFullParam, pPatchTargets->m_iNumTgts);
	aFitPatchShifts.DoIt(s_pPatchShifts, s_pLocalParam);
	delete s_pPatchShifts; s_pPatchShifts = 0L;
	//-----------------------------------------
	MrcUtil::CLocalAlignParam* pLocalParam = s_pLocalParam;
	s_pLocalParam = 0L;
	return pLocalParam;
}

void CPatchAlignMain::Run(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	this->Start();
}

void CPatchAlignMain::ThreadMain(void)
{	
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[m_iNthGpu]);
	//-------------------------------------------
	m_pLocalAlign = new CLocalAlign;
        m_pLocalAlign->Setup(s_pTomoStack, s_pFullParam, m_iNthGpu);
	//---------------------------------------------------------
	while(true)
	{	int iPatch = s_pNextItem->GetNext();
		if(iPatch < 0) break;
		else mAlignStack(iPatch);
	}
	delete m_pLocalAlign; m_pLocalAlign = 0L;
}

void CPatchAlignMain::mAlignStack(int iPatch)
{
	CPatchTargets* pPatchTargets = CPatchTargets::GetInstance();
	int iLeft = pPatchTargets->m_iNumTgts - 1 - iPatch;
	//-------------------------------------------------
	int aiCent[2] = {0};
	pPatchTargets->GetTarget(iPatch, aiCent);
	//-----------------------------------------
	MrcUtil::CAlignParam* pAlignParam = s_pFullParam->GetCopy();
	//----------------------------------------------------------
	printf("Align patch at (%d, %d), %d patches left\n", aiCent[0],
	   aiCent[1], iLeft);
	m_pLocalAlign->DoIt(s_pTomoStack, pAlignParam, aiCent);
	s_pPatchShifts->SetRawShift(iPatch, pAlignParam);
	//-----------------------------------------------
	CInput* pInput = CInput::GetInstance();
	char* pcLogFile = pInput->GetLogFile("PatchStack.txt", &iPatch);
	pAlignParam->LogShift(pcLogFile);
	if(pcLogFile != 0L) delete[] pcLogFile;
}

