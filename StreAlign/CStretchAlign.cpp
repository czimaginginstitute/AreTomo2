#include "CStreAlignInc.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace StreAlign;

static MrcUtil::CTomoStack* s_pTomoStack = 0L;
static MrcUtil::CAlignParam* s_pAlignParam = 0L;
static float s_fBFactor = 200.0f;
static float s_afBinning[] = {1.0f, 1.0f};

CStretchAlign::CStretchAlign(void)
{
}

CStretchAlign::~CStretchAlign(void)
{
}

float CStretchAlign::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	float fBFactor,
	float* pfBinning,
	int* piGpuIDs,
	int iNumGpus
)
{	s_pTomoStack = pTomoStack;
	s_pAlignParam = pAlignParam;
	s_fBFactor = fBFactor;
	s_afBinning[0] = pfBinning[0];
	s_afBinning[1] = pfBinning[1];
	//----------------------------
	Util::CNextItem nextItem;
	nextItem.Create(pTomoStack->m_aiStkSize[2]);
	//-------------------------------------------
	printf("Stretching based alignment\n");
	CStretchAlign* pThreads = new CStretchAlign[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	float fMaxErr = 0.0f;
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
		float fErr = pThreads[i].m_fMaxErr;
		if(fErr > fMaxErr) fMaxErr = fErr;
	}
	printf("       Max error: %.2f\n\n", fMaxErr);
	if(pThreads != 0L) delete[] pThreads;
	return fMaxErr; 
}

void CStretchAlign::Run
(	Util::CNextItem* pNextItem, 
	int iGpuID
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}

void CStretchAlign::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	m_stretchXcf.Setup(s_pTomoStack->m_aiStkSize, s_fBFactor); 
	m_fMaxErr = 0.0f;
	//---------------
	while(true)
	{	int iProj = m_pNextItem->GetNext();
		if(iProj < 0) break;
		float fErr = mMeasure(iProj);
		if(fErr > m_fMaxErr) m_fMaxErr = fErr;
	}
	m_stretchXcf.Clean();
}

float CStretchAlign::mMeasure(int iProj)
{
	int iRefProj = mFindRefIndex(iProj);
	if(iRefProj == iProj) return 0.0f;
	//--------------------------------
	float fRefTilt = s_pAlignParam->GetTilt(iRefProj);
	float fTilt = s_pAlignParam->GetTilt(iProj);
	float fTiltAxis = s_pAlignParam->GetTiltAxis(iProj);
	//--------------------------------------------
	float* pfRefProj = s_pTomoStack->GetFrame(iRefProj);
	float* pfProj = s_pTomoStack->GetFrame(iProj);
	m_stretchXcf.DoIt(pfRefProj, pfProj, fRefTilt, fTilt, fTiltAxis);
	//---------------------------------------------------------------
	float afShift[2] = {0.0f};
	m_stretchXcf.GetShift(s_afBinning[0], s_afBinning[1], afShift);
	s_pAlignParam->SetShift(iProj, afShift);
	printf("...... Proj %4d: %8.2f %8.2f  %8.2f\n", iProj+1,
		fTilt, afShift[0], afShift[1]); 
	//-------------------------------------
	float fErr = (float)sqrt(afShift[0] * afShift[0]
		+ afShift[1] * afShift[1]);
	return fErr;
}

int CStretchAlign::mFindRefIndex(int iProj)
{
	int iNumProjs = s_pTomoStack->m_aiStkSize[2];
	int iProj0 = iProj - 1;
	int iProj2 = iProj + 1;
	if(iProj0 < 0) iProj0 = iProj;
	if(iProj2 >= iNumProjs) iProj2 = iProj;
	//-------------------------------------
	double dTilt0 = fabs(s_pAlignParam->GetTilt(iProj0));
	double dTilt = fabs(s_pAlignParam->GetTilt(iProj));
	double dTilt2 = fabs(s_pAlignParam->GetTilt(iProj2));
	if(dTilt0 < dTilt) return iProj0;
	else if(dTilt2 < dTilt) return iProj2;
	else return iProj;
}
