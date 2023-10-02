#include "CTiltOffsetInc.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace TiltOffset;

CTiltOffsetMain1.cpp::CTiltOffsetMain(void)
{
	m_fD2R = 3.141592654f / 180.0f;
}

CTiltOffsetMain1.cpp::~CTiltOffsetMain(void)
{
	mClean();
}

float CTiltOffsetMain1.cpp::DoIt
(	MrcUtil::CAlignParam* pAlignParam,
	int iXcfBin
)
{	mClean();
	m_pAlignParam = pAlignParam;
	m_iNumFrames = m_pAlignParam->m_iNumFrames;
	m_pfShifts = new float[m_iNumFrames * 3];
	m_pfCos = m_pfShifts + m_iNumFrames;
	m_pfSin = m_pfShifts + m_iNumFrames * 2;
	//--------------------------------------
	m_fR = 0.0f;
	m_fA = 0.0f;
	mCalcShift();
	
}

void CTiltOffsetMain1::mCalcShift(void)
{
	float afShift[2] = {0.0f}, fTiltAxis = 0.0;
	for(int i=0; i<m_iNumFrames; i++)
	{	fTiltAxis = m_pAlignParam->GetTiltAxis(i) * m_fD2R;
		m_pAlignParam->GetShift(i, afShift);
		m_pfShifts[i] = (float)(afShift[0] * cos(fTiltAxis)
		   + afShift[1] * sin(fTiltAxis));
	}
}
		
void CTiltOffsetMain1::mCalcTrig(void)
{
	for(int i=0; i<m_iNumFrames; i++)
	{	float fTilt = (m_pAlignParam->GetTilt(i) + m_fA) * m_fD2R;
		m_pfCos = (float)cos(fTilt);
		m_pfSin = (float)sin(fTilt);
	}
}

void CTiltOffsetMain1::mCalcR(void)
{
	double dSumTop = 0, dSumBot = 0;
	float fCos0 = (float)cos(m_fA * m_fD2R);
	for(int i=0; i<m_iNumFrames; i++)
	{	float fCos = m_pfCos[i] - fCos0;
		dSumTop += (m_pfShifts[i] * fCos);
		dSumBot += (fCos * fCos);
	}
	m_fR = (float)(dSumTop / dSumBot);
}

void CTiltOffsetMain1::mCalcA(void)
{
	double dSumTop = 0, dSumBot = 0;
	for(int i=0; i<m_iNumFrames; i++)
	{	dSumTop += ((m_pfCos[i] - 1.0f) * m_pfSin[i]
		   + m_pfShifts[i] * m_pfSin[i] / m_fR);
		dSumBot += (m_pfSin[i] * m_pfSin[i]);
	}
	m_fA = (float)(dSumTop / dSumBot) / m_fD2R;
}
