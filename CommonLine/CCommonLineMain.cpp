#include "CCommonLineInc.h"
#include "../CInput.h"
#include <CuUtilFFT/GFFT1D.h>
#include <Util/Util_LinEqs.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace CommonLine;

CCommonLineMain::CCommonLineMain(void)
{
	m_pfFitAngles = 0L;
	m_iNumImgs = 0;
}

CCommonLineMain::~CCommonLineMain(void)
{
	this->Clean();
}

void CCommonLineMain::Clean(void)
{
	if(m_pfFitAngles != 0L) delete[] m_pfFitAngles;
	m_pfFitAngles = 0L;
}

float CCommonLineMain::DoInitial
(       MrcUtil::CTomoStack* pTomoStack,
        MrcUtil::CAlignParam* pAlignParam,
        float fAngRange,
        int iNumSteps
)
{	this->Clean();
	//------------
	CInput* pInput = CInput::GetInstance();
	CCommonLineParam* pCLParam = CCommonLineParam::GetInstance();
	pCLParam->Setup(fAngRange, iNumSteps, pTomoStack, pAlignParam);
	//-------------------------------------------------------------
	CPossibleLines* pPossibleLines = CGenLines::DoIt();
	CLineSet* pLineSet = new CLineSet;
	pLineSet->Setup();
	//----------------
	CFindTiltAxis findTiltAxis;
	float fRotAngle = findTiltAxis.DoIt(pPossibleLines, pLineSet);
	if(fRotAngle < -45 && pInput->m_afTiltAxis[0] == 0) 
	{	fRotAngle += 180.0f;
	}
	//--------------------------
	if(pPossibleLines != 0L) delete pPossibleLines;
	if(pLineSet != 0L) delete pLineSet;
	//---------------------------------
	printf("Initial estimate of tilt axes:\n");
	for(int i=0; i<pAlignParam->m_iNumFrames; i++)
	{	pAlignParam->SetTiltAxis(i, fRotAngle);
	}
	printf("New tilt axis: %.2f\n\n", fRotAngle);
	return findTiltAxis.m_fScore;
}

float CCommonLineMain::DoRefine
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	this->Clean();
	m_iNumImgs = pAlignParam->m_iNumFrames;
	//-------------------------------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance();
	pClParam->Setup(6.0f, 200, pTomoStack, pAlignParam);
	//--------------------------------------------------
	CPossibleLines* pPossibleLines = CGenLines::DoIt();
	CLineSet* pLineSet = new CLineSet;
	pLineSet->Setup();
	//----------------
	CRefineTiltAxis refineTiltAxis;
	refineTiltAxis.Setup(3, 10, 0.0001f);
	float fScore = refineTiltAxis.Refine(pPossibleLines, pLineSet);
	//-------------------------------------------------------------
	float* pfRotAngles = new float[pPossibleLines->m_iNumProjs];
	refineTiltAxis.GetRotAngles(pfRotAngles);
	//--------------------------------------
	if(pPossibleLines != 0L) delete pPossibleLines;
	if(pLineSet != 0L) delete pLineSet;
	//---------------------------------
	for(int i=0; i<m_iNumImgs; i++)
	{	pAlignParam->SetTiltAxis(i, pfRotAngles[i]);
	}
	if(pfRotAngles != 0L) delete[] pfRotAngles;
	return fScore;
}
