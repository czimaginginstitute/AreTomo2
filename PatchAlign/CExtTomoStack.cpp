#include "CPatchAlignInc.h"
#include <memory.h>
#include <math.h>
#include <stdio.h>

using namespace PatchAlign;

static MrcUtil::CTomoStack* s_pTomoStack = 0L;
static MrcUtil::CTomoStack* s_pPatchStack = 0L;
static int* s_piShifts = 0L;
static bool s_bRandomFill = true;

void CExtTomoStack::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CTomoStack* pPatchStack,
	int* piShifts,
	bool bRandomFill,
	int* piGpuIDs,
	int iNumGpus
)
{	s_pTomoStack = pTomoStack;
	s_pPatchStack = pPatchStack;
	s_piShifts = piShifts;
	s_bRandomFill = bRandomFill;
	//--------------------------
	Util::CNextItem nextItem;
	nextItem.Create(pTomoStack->m_aiStkSize[2]);
	//------------------------------------------
	CExtTomoStack* pThreads = new CExtTomoStack[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	//----------------
	s_pTomoStack = 0L;
	s_pPatchStack = 0L;
	s_piShifts = 0L;
}

CExtTomoStack::CExtTomoStack(void)
{
	m_gfRawProj = 0L;
	m_gfPatProj = 0L;
}

CExtTomoStack::~CExtTomoStack(void)
{
	this->Clean();
}

void CExtTomoStack::Clean(void)
{
	if(m_gfRawProj != 0L) cudaFree(m_gfRawProj);
	if(m_gfPatProj != 0L) cudaFree(m_gfPatProj);
	m_gfRawProj = 0L;
	m_gfPatProj = 0L;
}

void CExtTomoStack::Run
(	Util::CNextItem* pNextItem,
	int iGpuID
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}

void CExtTomoStack::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	int* piInSize = s_pTomoStack->m_aiStkSize;
	int iSize = piInSize[0] * piInSize[1];
	cudaMalloc(&m_gfRawProj, iSize * sizeof(float));
	//----------------------------------------------
	int* piPatSize = s_pPatchStack->m_aiStkSize;
	iSize = piPatSize[0] * piPatSize[1];
	cudaMalloc(&m_gfPatProj, iSize * sizeof(float));
	//----------------------------------------------
	bool bPadded = true;
	m_aGExtractPatch.SetSizes(piInSize, !bPadded, piPatSize, !bPadded);
	//-----------------------------------------------------------------
	while(true)
	{	int iProj = m_pNextItem->GetNext();
		if(iProj < 0) break;
		mExtractProj(iProj);
	}
	this->Clean();
}

void CExtTomoStack::mExtractProj(int iProj)
{
	int aiShift[2] = {0};
	aiShift[0] = -s_piShifts[iProj * 2];
	aiShift[1] = -s_piShifts[iProj * 2 + 1];
	//--------------------------------------
	float* pfProj = s_pTomoStack->GetFrame(iProj);
	size_t tBytes = sizeof(float) * s_pTomoStack->GetPixels();
	cudaMemcpy(m_gfRawProj, pfProj, tBytes, cudaMemcpyDefault);
	//---------------------------------------------------------
	m_aGExtractPatch.DoIt
	( m_gfRawProj, aiShift, s_bRandomFill, m_gfPatProj
	);
	//------------------------------------------------
	float* pfPatch = s_pPatchStack->GetFrame(iProj);
	tBytes = sizeof(float) * s_pPatchStack->GetPixels();
	cudaMemcpy(pfPatch, m_gfPatProj, tBytes, cudaMemcpyDefault);
}

