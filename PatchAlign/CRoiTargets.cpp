#include "CPatchAlignInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace PatchAlign;

static float s_fD2R = 0.0174533f;
CRoiTargets* CRoiTargets::m_pInstance = 0L;

CRoiTargets* CRoiTargets::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CRoiTargets;
	return m_pInstance;
}

void CRoiTargets::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CRoiTargets::CRoiTargets(void)
{
	m_iNumTgts = 0;
	//-----------------------------------------------------------
	// m_iTgtImg is the image where users select the targets.
	// This image must be the one collected at zero degree.
	//-----------------------------------------------------------
	m_iTgtImg = -1;
	m_piTargets = 0L;
}

CRoiTargets::~CRoiTargets(void)
{
	if(m_piTargets != 0L) delete[] m_piTargets;
}

void CRoiTargets::LoadRoiFile(void)
{
	CInput* pInput = CInput::GetInstance();
	Util::CReadDataFile aReadDataFile;
	bool bSuccess = aReadDataFile.DoIt(pInput->m_acRoiFile, 2);
	if(!bSuccess) return;
	//-------------------
	m_iNumTgts = aReadDataFile.m_iNumRows;
	if(m_iNumTgts <= 0) return;
	//------------------------------
	if(m_piTargets != 0L) delete[] m_piTargets;
	m_piTargets = new int[2 * m_iNumTgts];	
	//------------------------------------
	for(int i=0; i<m_iNumTgts; i++)
	{	int j =  2 * i;
		m_piTargets[j] = (int)aReadDataFile.GetData(i, 0);
		m_piTargets[j+1] = (int)aReadDataFile.GetData(i, 1);
	}
}

void CRoiTargets::SetTargetImage(MrcUtil::CAlignParam* pAlignParam)
{
	m_iTgtImg = pAlignParam->GetFrameIdxFromTilt(0.0f);
}

void CRoiTargets::MapToUntiltImage
(	MrcUtil::CAlignParam* pAlignParam,
	MrcUtil::CTomoStack* pTomoStack
)
{	if(m_iNumTgts <= 0) return;
	int iZeroTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
	if(iZeroTilt == m_iTgtImg) return;
	//----------------------------------
	float afGlobalShift[2] = {0.0f};
	pAlignParam->GetShift(m_iTgtImg, afGlobalShift);
	float fTiltAxis = pAlignParam->GetTiltAxis(m_iTgtImg);
	float fTgtTilt = pAlignParam->GetTilt(m_iTgtImg);
	float fRefTilt = pAlignParam->GetTilt(iZeroTilt);
	float fCosTgt = (float)cos(fTgtTilt * s_fD2R);
	float fCosRef = (float)cos(fRefTilt * s_fD2R);
	//---------------------------------------------------------
	float afS1[2] = {0.0f}, afS2[2] = {0.0f}, afS3[2] = {0.0f};
	float fCentX = pTomoStack->m_aiStkSize[0] * 0.5f;
	float fCentY = pTomoStack->m_aiStkSize[1] * 0.5f;
	//-----------------------------------------------
	printf("# ROI targets: map targets to new zero tilt image\n");
	printf("# image size: %d  %d\n", pTomoStack->m_aiStkSize[0],
	   pTomoStack->m_aiStkSize[1]);
	printf("# tgt tilt = %6.2f, ref tilt = %6.2f, tilt axis = %8.2f\n",
	   fTgtTilt, fRefTilt, fTiltAxis);
	printf("# tgt idx = %d, ref idx = %d\n", m_iTgtImg, iZeroTilt);
	//-------------------------------------------------------------- 
	for(int i=0; i<m_iNumTgts; i++)
	{	int j = i * 2;
		int k = j + 1;
		afS1[0] = m_piTargets[j] - fCentX - afGlobalShift[0];
		afS1[1] = m_piTargets[k] - fCentY - afGlobalShift[1];
		MrcUtil::CAlignParam::RotShift(afS1, -fTiltAxis, afS2);
		//-----------------------------------------------------
		afS2[0] *= (fCosRef / fCosTgt);
		MrcUtil::CAlignParam::RotShift(afS2, fTiltAxis, afS3);
		//----------------------------------------------------
		afS3[0] += fCentX;
		afS3[1] += fCentY;
		//----------------
		printf("  %4d  %8d  %8d  %8d  %8d\n", i+1, m_piTargets[j],
		   m_piTargets[k], (int)afS3[0], (int)afS3[1]);
		m_piTargets[j] = (int)afS3[0];
		m_piTargets[k] = (int)afS3[1];
	}
}

void CRoiTargets::GetTarget(int iTgt, int* piTgt)
{
	int i = 2 * iTgt;
	piTgt[0] = m_piTargets[i];
	piTgt[1] = m_piTargets[i+1];
}

int* CRoiTargets::GetTargets(bool bClean)
{
	int* piTargets = m_piTargets;
	if(bClean) 
	{	m_piTargets = 0L;
		m_iNumTgts = 0;
	}
	return piTargets;
}
