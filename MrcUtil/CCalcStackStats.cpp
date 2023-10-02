#include "CMrcUtilInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace MrcUtil;

static CTomoStack* s_pTomoStack = 0L;

void CCalcStackStats::DoIt
(       CTomoStack* pTomoStack,
	float* pfStats,
	int* piGpuIDs,
	int iNumGpus
)
{	s_pTomoStack = pTomoStack;
	//------------------------
	Util::CNextItem nextItem;
	int iAllFrms = pTomoStack->m_aiStkSize[2];
	nextItem.Create(iAllFrms);
	float* pfFrmStats = new float[iAllFrms * 4];
	//------------------------------------------
	CCalcStackStats* pThreads = new CCalcStackStats[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i], pfFrmStats);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	//----------------
	double dMean = 0.0, dStd = 0.0, dMin = 1e30, dMax = -1e30;
	float* pfMin = pfFrmStats;
	float* pfMax = pfFrmStats + iAllFrms;
	float* pfMean = pfFrmStats + iAllFrms * 2;
	float* pfMean2 = pfFrmStats + iAllFrms * 3;
	//-----------------------------------------
	for(int i=0; i<iAllFrms; i++)
	{	dMean += (pfMean[i] / iAllFrms);
		dStd += (pfMean2[i] / iAllFrms);
		if(dMax < pfMax[i]) dMax = pfMax[i];
		if(dMin > pfMin[i]) dMin = pfMin[i];
	}
	dStd = dStd - dMean * dMean;
	if(dStd < 0) dStd = 0.0;
	else dStd = sqrtf(dStd);
	if(pfFrmStats != 0L) delete[] pfFrmStats;
	//---------------------------------------
	pfStats[0] = (float)dMin;
	pfStats[1] = (float)dMax;
	pfStats[2] = (float)dMean;
	pfStats[3] = (float)dStd;
}


CCalcStackStats::CCalcStackStats(void)
{
}

CCalcStackStats::~CCalcStackStats(void)
{
}

void CCalcStackStats::Run
(	Util::CNextItem* pNextItem,
	int iGpuID,
	float* pfFrmStats
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	m_pfFrmStats = pfFrmStats;
	this->Start();
}

void CCalcStackStats::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	int iPixels = s_pTomoStack->GetPixels();
	size_t tBytes = sizeof(float) * iPixels;
	float *gfImg = 0L, *gfBuf = 0L;
	cudaMalloc(&gfImg, tBytes);
	cudaMalloc(&gfBuf, tBytes);
	//-------------------------
	bool bPadded = true;
	Util::GCalcMoment2D calcMoment2D;
	calcMoment2D.SetSize(s_pTomoStack->m_aiStkSize, !bPadded);
	Util::GFindMinMax2D findMinMax2D;
	findMinMax2D.SetSize(s_pTomoStack->m_aiStkSize, !bPadded);
	//--------------------------------------------------------
	float* pfMin = m_pfFrmStats;
	float* pfMax = m_pfFrmStats + s_pTomoStack->m_aiStkSize[2];
	float* pfMean = m_pfFrmStats + s_pTomoStack->m_aiStkSize[2] * 2;
	float* pfMean2 = m_pfFrmStats + s_pTomoStack->m_aiStkSize[2] * 3;
	//---------------------------------------------------------------
	while(true)
	{	int iFrm = m_pNextItem->GetNext();
		if(iFrm < 0) break;
		//-----------------
		float* pfFrame = s_pTomoStack->GetFrame(iFrm);
		cudaMemcpy(gfImg, pfFrame, tBytes, cudaMemcpyDefault);
		//----------------------------------------------------
		pfMin[iFrm] = findMinMax2D.DoMin(gfImg, true);
		pfMax[iFrm] = findMinMax2D.DoMax(gfImg, true);
		pfMean[iFrm] = calcMoment2D.DoIt(gfImg, 1, true);
		pfMean2[iFrm] = calcMoment2D.DoIt(gfImg, 2, true);	
	}
	if(gfImg != 0L) cudaFree(gfImg);
	if(gfBuf != 0L) cudaFree(gfBuf);
}
