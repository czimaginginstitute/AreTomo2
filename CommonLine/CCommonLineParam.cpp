#include "CCommonLineInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace CommonLine;

CCommonLineParam* CCommonLineParam::m_pInstance = 0L;

CCommonLineParam* CCommonLineParam::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CCommonLineParam;
	return m_pInstance;
}

void CCommonLineParam::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CCommonLineParam::CCommonLineParam(void)
{
	m_pfRotAngles = 0L;
}

CCommonLineParam::~CCommonLineParam(void)
{
	this->Clean();
}

void CCommonLineParam::Clean(void)
{
	if(m_pfRotAngles != 0L) delete[] m_pfRotAngles;
	m_pfRotAngles = 0L;
}

void CCommonLineParam::Setup
(	float fAngRange, 
	int iNumSteps,
	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	this->Clean();
	//------------
	m_fAngRange = fAngRange;
	m_iNumLines = iNumSteps;
	m_pTomoStack = pTomoStack;
	m_pAlignParam = pAlignParam;
	//--------------------------
	int iZeroTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	m_fTiltAxis = m_pAlignParam->GetTiltAxis(iZeroTilt);	
	//--------------------------------------------------
	m_pfRotAngles = new float[m_iNumLines];
	float fAngStep = fAngRange / (m_iNumLines - 1);
	m_pfRotAngles[0] = m_fTiltAxis - fAngStep * m_iNumLines / 2;
	for(int i=1; i<m_iNumLines; i++)
	{	m_pfRotAngles[i] = m_pfRotAngles[i-1] + fAngStep;
	}
	//-------------------------------------------------------
	int iSizeX = m_pTomoStack->m_aiStkSize[0];
	int iSizeY = m_pTomoStack->m_aiStkSize[1];
	double dRad = 3.1415926 / 180.0;
	float fMinSize = (float)1e20;
	for(int i=0; i<m_iNumLines; i++)
	{	float fSin = (float)fabs(sin(dRad * m_pfRotAngles[i]));
		float fCos = (float)fabs(cos(dRad * m_pfRotAngles[i]));
		float fSize1 = iSizeX / (fSin + 0.000001f);
		float fSize2 = iSizeY / (fCos + 0.000001f);
		if(fMinSize > fSize1) fMinSize = fSize1;
		if(fMinSize > fSize2) fMinSize = fSize2;
	}
	m_iLineSize = (int)fMinSize;
	m_iLineSize = m_iLineSize / 2 * 2 - 100;
	m_iCmpLineSize = m_iLineSize / 2 + 1;
	printf("CommonLine: Line Size = %d\n", m_iLineSize);
}
