#include "CProjAlignInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace ProjAlign;

static MrcUtil::CTomoStack* s_pTomoStack = 0L;

void CRemoveSpikes::DoIt
(       MrcUtil::CTomoStack* pTomoStack,
	int* piGpuIDs,
	int iNumGpus
)
{	s_pTomoStack = pTomoStack;
	//------------------------
	Util::CNextItem nextItem;
	nextItem.Create(s_pTomoStack->m_aiStkSize[2]);
	//--------------------------------------------
	CRemoveSpikes* pThreads = new CRemoveSpikes[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
}


CRemoveSpikes::CRemoveSpikes(void)
{
}

CRemoveSpikes::~CRemoveSpikes(void)
{
}

void CRemoveSpikes::Run
(	Util::CNextItem* pNextItem,
	int iGpuID
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}

void CRemoveSpikes::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	bool bPadded = true;
	bool bGpu = true;
	int iWinSize = 11;
	m_tFmBytes = sizeof(float) * s_pTomoStack->GetPixels();
	cudaMalloc(&m_gfInFrm, m_tFmBytes);
	cudaMalloc(&m_gfOutFrm, m_tFmBytes);
	Util::GRemoveSpikes2D removeSpikes;
	//----------------------------------
	while(true)
	{	int y = m_pNextItem->GetNext();
		if(y < 0) break;
		//--------------
		float* pfFrm = s_pTomoStack->GetFrame(y);
		cudaMemcpy(m_gfInFrm, pfFrm, m_tFmBytes, cudaMemcpyDefault);
		removeSpikes.DoIt(m_gfInFrm, s_pTomoStack->m_aiStkSize,
			!bPadded, iWinSize, m_gfOutFrm);
		cudaMemcpy(pfFrm, m_gfOutFrm, m_tFmBytes, cudaMemcpyDefault);
	}
	//-------------------------------------------------------------------
	cudaFree(m_gfInFrm);
	cudaFree(m_gfOutFrm);
}
