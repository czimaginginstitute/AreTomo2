#include "CTiltOffsetInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace TiltOffset;

CTiltOffsetMain::CTiltOffsetMain(void)
{
	m_pCorrTomoStack = 0L;
}

CTiltOffsetMain::~CTiltOffsetMain(void)
{
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = 0L;
	m_aStretchCC2D.Clean();
}

void CTiltOffsetMain::Setup(int* piStkSize, int iXcfBin, int iNthGpu)
{
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[iNthGpu]);
	//-----------------------------------------	
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	bool bShiftOnly = true, bRandomFill = true;
	bool bFourierCrop = true, bRWeight = true;
	m_pCorrTomoStack = new Correct::CCorrTomoStack;
	m_pCorrTomoStack->Set0(pInput->m_piGpuIDs[iNthGpu]);
	m_pCorrTomoStack->Set1(piStkSize, 0, 0.0f);
	m_pCorrTomoStack->Set2((float)iXcfBin, !bFourierCrop, !bRandomFill);
	m_pCorrTomoStack->Set3(bShiftOnly, false, !bRWeight);
	//---------------------------------------------------
	bool bClean = true;
	m_pTomoStack = m_pCorrTomoStack->GetCorrectedStack(!bClean);
	//----------------------------------------------------------
	bool bPadded = true;
	m_aStretchCC2D.SetSize(m_pTomoStack->m_aiStkSize, !bPadded);
}
	

float CTiltOffsetMain::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	m_pAlignParam = pAlignParam;
	//--------------------------
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[0]);
	//-----------------------------------
	bool bClean = true;
	m_pCorrTomoStack->DoIt(pTomoStack, m_pAlignParam, 0L);
	m_pTomoStack = m_pCorrTomoStack->GetCorrectedStack(!bClean);
	//----------------------------------------------------------
	printf("Determine tilt angle offset.\n");
	float fBestOffset = mSearch(31, 1.0f, 0.0f);
	//------------------------------------------
	m_aStretchCC2D.Clean();
	return fBestOffset;
}

float CTiltOffsetMain::mSearch
(	int iNumSteps, 
	float fStep, 
	float fInitOffset
)
{	float fMaxCC = 0.0f;
        float fBestOffset = 0.0f;
        for(int i=0; i<iNumSteps; i++)
        {       float fOffset = fInitOffset + (i - iNumSteps / 2) * fStep;
                float fCC = mCalcAveragedCC(fOffset);
                if(fCC > fMaxCC)
                {       fMaxCC = fCC;
                        fBestOffset = fOffset;
                }
                printf("...... %8.2f  %.4e\n", fOffset, fCC);
        }
	printf("Tilt offset: %8.2f,  CC: %.4f\n\n", fBestOffset, fMaxCC);
	return fBestOffset;
}

float CTiltOffsetMain::mCalcAveragedCC(float fTiltOffset)
{
	m_pAlignParam->AddTiltOffset(fTiltOffset);
	//----------------------------------------
	int iCount = 0;
	float fCCSum = 0.0f;
	int iZeroTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	for(int i=0; i<m_pAlignParam->m_iNumFrames; i++)
	{	if(i == iZeroTilt) continue;
		int iRefTilt = (i < iZeroTilt) ? i+1 : i-1;
		float fCC = mCorrelate(iRefTilt, i);
		fCCSum += fCC;
		iCount++;
	}
	float fMeanCC = fCCSum / iCount;
	//------------------------------
	m_pAlignParam->AddTiltOffset(-fTiltOffset);
	return fMeanCC;
}

float CTiltOffsetMain::mCorrelate(int iRefProj, int iProj)
{
	float* pfRefProj = m_pTomoStack->GetFrame(iRefProj);
	float* pfProj = m_pTomoStack->GetFrame(iProj);
	float fRefTilt = m_pAlignParam->GetTilt(iRefProj);
	float fTilt = m_pAlignParam->GetTilt(iProj);
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	//--------------------------------------------------
	float fCC = m_aStretchCC2D.DoIt
	( pfRefProj, pfProj, fRefTilt, fTilt, fTiltAxis
	);
	return fCC;
}

