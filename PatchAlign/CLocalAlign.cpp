#include "CPatchAlignInc.h"
#include "../CInput.h"
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

CLocalAlign::CLocalAlign(void)
{
	m_iZeroTilt = -1;
	m_pProjAlignMain = 0L;
}

CLocalAlign::~CLocalAlign(void)
{
	if(m_pProjAlignMain != 0L) delete m_pProjAlignMain;
	m_pProjAlignMain = 0L;
}

void CLocalAlign::Setup
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	int iNthGpu
)
{	if(m_pProjAlignMain != 0L) delete m_pProjAlignMain;
	CInput* pInput = CInput::GetInstance();
	m_pProjAlignMain = new ProjAlign::CProjAlignMain;
	m_pProjAlignMain->Setup(pTomoStack, pAlignParam, 
	   pInput->m_afBFactor[1], iNthGpu);
	//----------------------------------
	m_iZeroTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
}

void CLocalAlign::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	int* piRoi
)
{	pAlignParam->ResetShift();
	//------------------------
	float afS1[2] = {0.0f}, afS2[2] = {0.0f}, afS3[2] = {0.0f};
	afS1[0] = piRoi[0] - pTomoStack->m_aiStkSize[0] * 0.5f;
	afS1[1] = piRoi[1] - pTomoStack->m_aiStkSize[1] * 0.5f;
	float fCosRef = (float)cos(pAlignParam->GetTilt(m_iZeroTilt) * s_fD2R);
	for(int i=0; i<pTomoStack->m_aiStkSize[2]; i++)
	{	float fTilt = pAlignParam->GetTilt(i) * s_fD2R;
		float fTiltAxis = pAlignParam->GetTiltAxis(i);
		MrcUtil::CAlignParam::RotShift(afS1, -fTiltAxis, afS2);
		afS2[0] *= (float)(cos(fTilt) / fCosRef);
		MrcUtil::CAlignParam::RotShift(afS2, fTiltAxis, afS3);
		pAlignParam->SetShift(i, afS3);
	}
	//-------------------------------------
	CInput* pInput = CInput::GetInstance();
	ProjAlign::CParam* pParam = ProjAlign::CParam::GetInstance();
	pParam->m_iVolZ = pInput->m_iAlignZ;
	pParam->m_fXcfSize = 1024.0f * 1.5f;
	pParam->m_afMaskSize[0] = 0.8f;
	pParam->m_afMaskSize[1] = 0.8f;
	//-----------------------------
	pAlignParam->SetRotationCenterZ(0.0f);
	float fLastErr = m_pProjAlignMain->DoIt(pTomoStack, pAlignParam);
	MrcUtil::CAlignParam* pLastParam = pAlignParam->GetCopy();
	//--------------------------------------------------------
	pParam->m_fXcfSize = 1024.0f * 2.0f;
	pParam->m_afMaskSize[0] = 0.125f;
	pParam->m_afMaskSize[1] = 0.125f;
	//-------------------------------
	int iIterations = 2;
	int iLastIter = iIterations - 1;
	for(int i=0; i<iIterations; i++)
	{	float fErr = m_pProjAlignMain->DoIt(pTomoStack, pAlignParam);
		//if(fErr < 2.0f) break;
		//--------------------
		pParam->m_afMaskSize[0] = 0.1f;
		pParam->m_afMaskSize[1] = 0.1f;
		//-----------------------------	
		if(fErr < fLastErr)
		{	pLastParam->Set(pAlignParam);
			if((fLastErr - fErr) < 1) break;
			else fLastErr = fErr; 
		}
		else
		{	pAlignParam->Set(pLastParam);
			break;
		}
	}
	delete pLastParam;
}

