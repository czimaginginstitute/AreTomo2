#include "CPatchAlignInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace PatchAlign;

static float s_fD2R = 0.0174533f;
CPatchTargets* CPatchTargets::m_pInstance = 0L;

CPatchTargets* CPatchTargets::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CPatchTargets;
	return m_pInstance;
}

void CPatchTargets::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CPatchTargets::CPatchTargets(void)
{
	m_iNumTgts = 0;
	//-----------------------------------------------------------
	// m_iTgtImg is the image where the targets are selected. It
	//-----------------------------------------------------------
	m_iTgtImg = -1;
	m_piTargets = 0L;
}

CPatchTargets::~CPatchTargets(void)
{
	this->Clean();
}

void CPatchTargets::Clean(void)
{
	if(m_piTargets != 0L) delete[] m_piTargets;
	m_piTargets = 0L;
}

void CPatchTargets::DetectTargets
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	bool bClean = true;
	this->Clean();
	//-----------------
	CRoiTargets* pRoiTargets = CRoiTargets::GetInstance();
	if(pRoiTargets->m_iNumTgts > 0)
	{	m_iNumTgts = pRoiTargets->m_iNumTgts;
		m_piTargets = pRoiTargets->GetTargets(bClean);
		CRoiTargets::DeleteInstance();
		return;
	}
	//-----------------------------------------------------------------
	CInput* pInput = CInput::GetInstance();
	m_iNumTgts = pInput->m_aiNumPatches[0] * pInput->m_aiNumPatches[1];
	if(m_iNumTgts <= 0) return;
	//-------------------------
	int iStkAxis = (pTomoStack->m_aiStkSize[0] > pTomoStack->m_aiStkSize[1])
	   ? 1 : 2;
	int iPatAxis = (pInput->m_aiNumPatches[0] > pInput->m_aiNumPatches[1])
	   ? 1 : 2;
	if(iStkAxis != iPatAxis)
	{	int iPatchesX = pInput->m_aiNumPatches[0];
		pInput->m_aiNumPatches[0] = pInput->m_aiNumPatches[1];
		pInput->m_aiNumPatches[1] = iPatchesX;
	}
	//------------------------------------------------------------
	m_iTgtImg = pAlignParam->GetFrameIdxFromTilt(0.0f);
	m_piTargets = new int[m_iNumTgts * 2];
	//------------------------------------
	CDetectFeatures* pDetectFeatures = CDetectFeatures::GetInstance();
	pDetectFeatures->SetSize(pTomoStack->m_aiStkSize,
	   pInput->m_aiNumPatches);
	m_iTgtImg = pAlignParam->GetFrameIdxFromTilt(0.0f);
	float* pfZeroImg = pTomoStack->GetFrame(m_iTgtImg);
	pDetectFeatures->DoIt(pfZeroImg);
	//---------------------------------------------------
	printf("# Patch alignment: targets detected automatically\n");
	printf("# Image size: %d  %d\n", pTomoStack->m_aiStkSize[0],
	   pTomoStack->m_aiStkSize[1]);
	printf("# Number of patches: %d  %d\n", pInput->m_aiNumPatches[0],
	   pInput->m_aiNumPatches[1]);
	//----------------------------------------------------------------
        int iPatX = pTomoStack->m_aiStkSize[0] / pInput->m_aiNumPatches[0];
        int iPatY = pTomoStack->m_aiStkSize[1] / pInput->m_aiNumPatches[1];
        for(int y=0; y<pInput->m_aiNumPatches[1]; y++)
        {       int iCentY = y * iPatY + iPatY / 2;
                for(int x=0; x<pInput->m_aiNumPatches[0]; x++)
                {       int iCentX = x * iPatX + iPatX / 2;
                        int iPatch = y * pInput->m_aiNumPatches[0] + x;
			int* piTgt = m_piTargets + 2 * iPatch;
                        pDetectFeatures->GetCenter(iPatch, piTgt);
                        printf("%3d %6d %6d %6d %6d\n", iPatch + 1,
                           iCentX, iCentY, piTgt[0], piTgt[1]);
                }
        }
        printf("\n");
	CDetectFeatures::DeleteInstance();
}

void CPatchTargets::GetTarget(int iTgt, int* piTgt)
{
	piTgt[0] = m_piTargets[iTgt * 2];
	piTgt[1] = m_piTargets[iTgt * 2 + 1];
}
