#include "CTiltOffsetInc.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace TiltOffset;

CTiltAxisMain::CTiltAxisMain(void)
{
	m_pTomoStack = 0L;
	m_gfPadProj1 = 0L;
	m_gfPadProj2 = 0L;
}

CTiltAxisMain::~CTiltAxisMain(void)
{
	this->Clean();
}

void CTiltAxisMain::Clean(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	m_pTomoStack = 0L;
	//----------------
	if(m_gfPadProj1 != 0L) cudaFree(m_gfPadProj1);
	if(m_gfPadProj2 != 0L) cudaFree(m_gfPadProj2);
	m_gfPadProj1 = 0L;
	m_gfPadProj2 = 0L;
	//----------------
	m_fft2D.DestroyPlan();
}

void CTiltAxisMain::Setup
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	float fBFactor,
	int iXcfBin
)
{	this->Clean();
	//------------
	m_pAlignParam = pAlignParam;
	m_fBFactor = fBFactor;
	m_iXcfBin = iXcfBin;
	//------------------
	bool bShiftOnly = true;
	bool bRandomFill = true;
	bool bFourierCrop = true;
	CInput* pInput = CInput::GetInstance();
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	m_pTomoStack = Correct::CCorrTomoStack::DoIt
	( pTomoStack, pAlignParam, 0L, 
	  m_iXcfBin, bShiftOnly, !bRandomFill,
	  !bFourierCrop, pInput->m_piGpuIDs, pInput->m_iNumGpus
	);
	//--------------------------------------
	m_aiPadSize[0] = (m_pTomoStack->m_aiStkSize[0] / 2 + 1) * 2;
	m_aiPadSize[1] = m_pTomoStack->m_aiStkSize[1];
	size_t tBytes = sizeof(float) * m_aiPadSize[0] * m_aiPadSize[1];
	cudaMalloc(&m_gfPadProj1, tBytes);
	cudaMalloc(&m_gfPadProj2, tBytes);
	//--------------------------------
	bool bForward = true;
	m_fft2D.CreatePlan(m_pTomoStack->m_aiStkSize, bForward);
}

float CTiltAxisMain::Measure(float fTiltAxis)
{
	float fCCSum = 0.0f;
	int iZeroTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	for(int i=iZeroTilt+1; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	float fCC = mCorrelate(i, -1, fTiltAxis);
		fCCSum += fCC;
	}
	for(int i=0; i<iZeroTilt; i++)
	{	float fCC = mCorrelate(i, 1, fTiltAxis);
		fCCSum += fCC;
	}
	float fScore = fCCSum / (m_pTomoStack->m_aiStkSize[2] - 1);
	return fScore;
}

float CTiltAxisMain::mCorrelate(int iProj, int iStep, float fTiltAxis)
{	/*
	int iRefProj = iProj + iStep;
	mGetGProj(iProj, m_gfPadProj1);
	mGetGProj(iRefProj, m_gfPadProj2);
	//--------------------------------
	int* piProjSize = m_pTomoStack->m_aiStkSize;
	float fTilt = m_pAlignParam->GetTilt(iProj);
	float fRefTilt = m_pAlignParam->GetTilt(iRefProj);
	//------------------------------------------------
	Util::GStretch aGStretch;
        double dRad = 4 * atan(1.0) / 180.0f;
        float fCos = (float)cos(dRad * fTilt);
        float fStretch = (float)(cos(dRad * fRefTilt) / cos(dRad * fTilt));
	aGStretch.SetInSize(piProjSize, !bPadded);
	aGStretch.DoIt(m_gfPadProj1, fStretch, fTiltAxis, 
		m_gfPadProj1, !bPadded);
	//------------------------------
	Util::GRealCC2D aGCC2D;
	float fCC = aGCC2D.DoIt
	( (cufftComplex*)m_gfPadProj1, 
	  (cufftComplex*)m_gfPadProj2, aiCmpSize
	);
	return fCC;*/
	return 0.0f;
}

void CTiltAxisMain::mGetGProj(int iProj, float* gfPadProj)
{
	size_t tBytes = sizeof(float) * m_pTomoStack->m_aiStkSize[0];
	float* pfProj = m_pTomoStack->GetFrame(iProj);
	for(int y=0; y<m_aiPadSize[1]; y++)
	{	float* pfSrc = pfProj + y * m_pTomoStack->m_aiStkSize[0];
		float* gfDst = gfPadProj + y * m_aiPadSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	} 
}
