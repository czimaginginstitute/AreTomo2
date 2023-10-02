#include "CStreAlignInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace StreAlign;

CStreAlignMain::CStreAlignMain(void)
{
	m_pCorrTomoStack = 0L;
	m_pMeaParam = 0L;
}

CStreAlignMain::~CStreAlignMain(void)
{
	this->Clean();
}

void CStreAlignMain::Clean(void)
{
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	if(m_pMeaParam != 0L) delete m_pMeaParam;
	m_pCorrTomoStack = 0L;
	m_pMeaParam = 0L;
}

void CStreAlignMain::Setup
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	this->Clean();
	//------------
	int iBinX = (int)(pTomoStack->m_aiStkSize[0] / 1024.0f + 0.1f);
	int iBinY = (int)(pTomoStack->m_aiStkSize[1] / 1024.0f + 0.1f);
	int iBin = (iBinX < iBinY) ? iBinX : iBinY;
	if(iBin < 1) iBin = 1;
	//--------------------
	CInput* pInput = CInput::GetInstance();
	m_pCorrTomoStack = new Correct::CCorrTomoStack;
	bool bShiftOnly = true, bRandFill = true;
	bool bFourierCrop = true, bRWeight = true;
	m_pCorrTomoStack->Set0(pInput->m_piGpuIDs[0]);
	m_pCorrTomoStack->Set1(pTomoStack->m_aiStkSize, 0, 0.0f);
	m_pCorrTomoStack->Set2((float)iBin, !bFourierCrop, bRandFill);
	m_pCorrTomoStack->Set3(bShiftOnly, false, !bRWeight);
	//------------------------------------------------------------
	m_pBinStack = m_pCorrTomoStack->GetCorrectedStack(false);
	m_afBinning[0] = pTomoStack->m_aiStkSize[0] * 1.0f
	   / m_pBinStack->m_aiStkSize[0];
	m_afBinning[1] = pTomoStack->m_aiStkSize[1] * 1.0f
	   / m_pBinStack->m_aiStkSize[1];
}	

void CStreAlignMain::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[0]);
	//-----------------------------------
	m_pTomoStack = pTomoStack;
	m_pAlignParam = pAlignParam;
	//--------------------------
	m_pMeaParam = m_pAlignParam->GetCopy();
	m_pMeaParam->ResetShift();
	//------------------------
	printf("Pre-align tilt series\n");
	m_pCorrTomoStack->DoIt(m_pTomoStack, m_pAlignParam, 0L);
	float fErr = mMeasure();
	mUpdateShift();
        printf("Error: %8.2f\n\n", fErr);
	//-------------------------------
	delete m_pMeaParam; m_pMeaParam = 0L;
}

float CStreAlignMain::mMeasure(void)
{
	CInput* pInput = CInput::GetInstance();
	m_pMeaParam->ResetShift();
	//------------------------
	float fMaxErr = CStretchAlign::DoIt(m_pBinStack, m_pMeaParam, 400.0, 
	   m_afBinning, pInput->m_piGpuIDs, pInput->m_iNumGpus);
	//---------------------------------------------------
	return fMaxErr;
}

void CStreAlignMain::mUpdateShift(void)
{
	int iZeroTilt = m_pMeaParam->GetFrameIdxFromTilt(0.0f);
	m_pMeaParam->MakeRelative(iZeroTilt);
	//-----------------------------------
	float afSumShift[2] = {0.0f};
	float afShift[2] = {0.0f};
	for(int i=iZeroTilt-1; i>=0; i--)
	{	m_pMeaParam->GetShift(i, afShift);
		afSumShift[0] += afShift[0];
		afSumShift[1] += afShift[1];
		mUnstretch(i+1, i, afSumShift);
		m_pAlignParam->AddShift(i, afSumShift);
	}
	//---------------------------------------------
	memset(afSumShift, 0, sizeof(afSumShift));
	for(int i=iZeroTilt+1; i<m_pMeaParam->m_iNumFrames; i++)
        {       m_pMeaParam->GetShift(i, afShift);
                afSumShift[0] += afShift[0];
                afSumShift[1] += afShift[1];
                mUnstretch(i-1, i, afSumShift);
                m_pAlignParam->AddShift(i, afSumShift);
        }
}

void CStreAlignMain::mUnstretch(int iLowTilt, int iHighTilt, float* pfShift)
{
        double dRad = 3.14159 / 180.0;
        float fLowTilt = m_pAlignParam->GetTilt(iLowTilt);
        float fHighTilt = m_pAlignParam->GetTilt(iHighTilt);
        float fTiltAxis = m_pAlignParam->GetTiltAxis(iHighTilt);
        double dStretch = cos(fLowTilt * dRad) / cos(fHighTilt * dRad);
        //-------------------------------------------------------------
        Util::GStretch aGStretch;
        aGStretch.CalcMatrix((float)dStretch, fTiltAxis);
        aGStretch.Unstretch(pfShift, pfShift);
}
