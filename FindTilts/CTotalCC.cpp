#include "CFindTiltsInc.h"
#include "../Util/CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>

using namespace FindTilts;

static float** s_ppfProjs = 0L;
static float* s_pfTilts = 0L;
static float* s_pfTiltAxes = 0L;
static int s_aiProjSize[3] = {0};

float CTotalCC::DoIt
(	float** ppfProjs,
	float* pfTilts,
	float* pfTiltAxes,
	int* piProjSize,
	int* piGpuIDs,
	int iNumGpus
)
{	s_ppfProjs = ppfProjs;
	s_pfTilts = pfTilts;
	s_pfTiltAxes = pfTiltAxes;
	memcpy(s_aiProjSize, piProjSize, sizeof(int) * 3);
	//------------------------------------------------
	Util::CNextItem nextItem;
	nextItem.Create(s_aiProjSize[2]);
	//-------------------------------
	float fCCSum = 0.0f;
	float fStdSum = 0.0f;
	//-------------------
	CTotalCC* pThreads = new CTotalCC[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	};
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
		fCCSum += pThreads[i].m_fCCSum;
		fStdSum += pThreads[i].m_fStdSum;
	}
	float fCC = 0.0f;
	if(fStdSum > 0) fCC = fCCSum / fStdSum;
	//-------------------------------------
	if(pThreads != 0L) delete[] pThreads;
	return fCC;
}

CTotalCC::CTotalCC(void)
{
	m_fCCSum = 0.0f;
	m_fStdSum = 0.0f;
}

CTotalCC::~CTotalCC(void)
{
}

void CTotalCC::Run
(	Util::CNextItem* pNextItem,
   	int iGpuID
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}

void CTotalCC::ThreadMain(void)
{
	bool bGpu = true;
	cudaSetDevice(m_iGpuID);
	//----------------------
	m_stretchCC.Setup(s_aiProjSize, 4, 200.0f);
	m_fCCSum = 0.0f;
	m_fStdSum = 0.0f;
	//---------------
	float afTilt[2] = {0.0f};
	float fTiltAxis = 0.0f;
	//---------------------
	while(true)
	{	int iProj = m_pNextItem->GetNext();
		if(iProj < 0 || iProj == (s_aiProjSize[2] - 1)) break;
		//----------------------------------------------------
		float* pfProj1 = s_ppfProjs[iProj];
		float* pfProj2 = s_ppfProjs[iProj+1];
		afTilt[0] = s_pfTilts[iProj];
		afTilt[1] = s_pfTilts[iProj + 1];
		fTiltAxis = s_pfTiltAxes[iProj];
		//------------------------------
		m_stretchCC.DoIt(pfProj1, pfProj2, afTilt, fTiltAxis);
		m_fCCSum += m_stretchCC.m_fCCSum;
		m_fStdSum += m_stretchCC.m_fStdSum;
	}
	//-----------------------------------------
	m_stretchCC.Clean();
}

