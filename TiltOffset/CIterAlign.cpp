#include "CStreAlignInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace Strealign;

CIterAlign::CIterAlign(void)
{
}

CIterAlign::~CIterAlign(void)
{
}

void CIterAlign::DoIt
(	::MrcUtil::CTomoStack* pTomoStack,
	::Correct::CStackShift* pStackShift
)
{	m_pTomoStack = pTomoStack;
	m_pStackShift = pStackShift;
	//--------------------------
	CParam* pParam = CParam::GetInstance();
	float fErr = 10000.0f;
	int iIters = 0;	
	for(int i=0; i<pParam->m_iIterations; i++)
        {       iIters = i + 1;
		fErr = mMeasure(i);
		if(fErr < pParam->m_fTol) break;
        }
        printf("Total Iterations: %2d  Error: %8.2f\n\n", iIters, fErr);
}

float CIterAlign::mMeasure(int iIter)
{
	CParam* pParam = CParam::GetInstance();
	CStretchAlign aStretchAlign;
	float fMaxErr = aStretchAlign.DoIt
	(  m_pTomoStack, m_pStackShift,
	   pParam->m_fBFactor, 1
	);
	Correct::CStackShift* pNewShift = aStretchAlign.GetShift(true);
	//----------------------------------------------------
	float afShift[2];
	for(int i=0; i<m_pStackShift->m_iNumFrames; i++)
	{	pNewShift->GetShift(i, afShift);
		float fTilt = pNewShift->GetTilt(i);
		float fTiltAxis = pNewShift->GetTiltAxis(i);
		m_pStackShift->SetTilt(i, fTilt);
		m_pStackShift->SetTiltAxis(i, fTiltAxis);
		//---------------------------------------
		m_pStackShift->AddShift(i, afShift);
	}
	return fMaxErr;
}
