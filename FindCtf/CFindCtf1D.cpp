#include "CFindCtfInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace FindCtf;

CFindCtf1D::CFindCtf1D(void)
{
	m_pFindDefocus1D = 0L;
	m_gfRadialAvg = 0L;
}

CFindCtf1D::~CFindCtf1D(void)
{
	this->Clean();
}

void CFindCtf1D::Clean(void)
{
	if(m_pFindDefocus1D != 0L) 
	{	delete m_pFindDefocus1D;
		m_pFindDefocus1D = 0L;
	}
	if(m_gfRadialAvg != 0L)
	{	cudaFree(m_gfRadialAvg);
		m_gfRadialAvg = 0L;
	}
	CFindCtfBase::Clean();
}

void CFindCtf1D::Setup1(CCtfTheory* pCtfTheory)
{
	this->Clean();
	CFindCtfBase::Setup1(pCtfTheory);
	cudaMalloc(&m_gfRadialAvg, sizeof(float) * m_aiCmpSize[0]);
	//---------------------------------------------------------
	m_pFindDefocus1D = new CFindDefocus1D;
	CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
	m_pFindDefocus1D->Setup(pCtfParam, m_aiCmpSize[0]);
	//-------------------------------------------------
	m_pFindDefocus1D->SetResRange(m_afResRange);
}

void CFindCtf1D::Do1D(void)
{	
	mCalcRadialAverage();
	mFindDefocus();
	//-------------
	float fDfRange = fmaxf(0.3f * m_fDfMin, 3000.0f); 
	mRefineDefocus(fDfRange);
}

void CFindCtf1D::Refine1D(float fInitDf, float fDfRange)
{
	m_fDfMin = fInitDf;
	m_fDfMax = fInitDf;
	m_fScore = (float)-1e20;
	//----------------------
	mCalcRadialAverage();
	mRefineDefocus(fDfRange);
}

void CFindCtf1D::mFindDefocus(void)
{
	CInput* pInput = CInput::GetInstance();
	float fPixSize2 = pInput->m_fPixelSize * pInput->m_fPixelSize;
	float afDfRange[2] = {0.0f};
	afDfRange[0] = 3000.0f * fPixSize2;
	afDfRange[1] = 30000.0f * fPixSize2;
	//------------------
	float afPhaseRange[2] = {0.0f};
	afPhaseRange[0] = fmax(m_fExtPhase - m_fPhaseRange / 2, 0.0f);
	afPhaseRange[1] = fmin(afPhaseRange[0] + m_fPhaseRange, 180.0f);
	//------------------
	m_pFindDefocus1D->DoIt(afDfRange, afPhaseRange, m_gfRadialAvg);
	m_fExtPhase = m_pFindDefocus1D->m_fBestPhase;
	m_fDfMin = m_pFindDefocus1D->m_fBestDf;
	m_fDfMax = m_fDfMin;
	m_fScore = m_pFindDefocus1D->m_fMaxCC;
}

void CFindCtf1D::mRefineDefocus(float fDfRange)
{
	CInput* pInput = CInput::GetInstance();
	float fPixSize2 = pInput->m_fPixelSize * pInput->m_fPixelSize;
	float afDfRange[2] = {0.0f};
	afDfRange[0] = m_fDfMin; // central value
	afDfRange[1] = fDfRange * fPixSize2; 
	//----------------------------------
	float afPhaseRange[2] = {0.0f};
	afPhaseRange[0] = m_fExtPhase;
	afPhaseRange[1] = m_fPhaseRange * 0.2f;
	if(m_fPhaseRange > 0) 
	{	afPhaseRange[1] = fmaxf(afPhaseRange[1], 4.0f);
	}
	//-----------------------------------------------------
	m_pFindDefocus1D->DoIt(afDfRange, afPhaseRange, m_gfRadialAvg);
	m_fExtPhase = m_pFindDefocus1D->m_fBestPhase;
	m_fDfMin = m_pFindDefocus1D->m_fBestDf;
	m_fDfMax = m_fDfMin;
	m_fScore = m_pFindDefocus1D->m_fMaxCC;
}

void CFindCtf1D::mCalcRadialAverage(void)
{
	GRadialAvg aGRadialAvg;
	aGRadialAvg.DoIt(m_gfCtfSpect, m_gfRadialAvg, m_aiCmpSize);
}

