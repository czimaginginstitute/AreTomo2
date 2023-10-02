#include "CMrcUtilInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace MrcUtil;

static CTomoStack* s_pTomoStack = 0L;
static CAlignParam* s_pAlignParam = 0L;
static float s_fThreshold = 0.7f;

void CRemoveDarkFrames::DoIt
(       CTomoStack* pTomoStack,
	CAlignParam* pAlignParam,
	float fThreshold,
	int* piGpuIDs,
	int iNumGpus
)
{	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
        pDarkFrames->Setup(pTomoStack->m_aiStkSize);
	if(fThreshold <= 0 || fThreshold >= 1) return;
	//--------------------------------------------
	s_pTomoStack = pTomoStack;
	s_pAlignParam = pAlignParam;
	s_fThreshold = fThreshold;
	//------------------------
	Util::CNextItem nextItem;
	int iAllFrms = pTomoStack->m_aiStkSize[2];
	nextItem.Create(iAllFrms);
	float* pfMeans = new float[iAllFrms];
	float* pfStds = new float[iAllFrms];
	//----------------------------------
	CRemoveDarkFrames* pThreads = new CRemoveDarkFrames[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i], pfMeans, pfStds);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	//----------------
	printf("# index  tilt    mean         std      ratio\n");
	for(int i=0; i<iAllFrms; i++)
	{	float fMean = (float)fabs(pfMeans[i]);
		float fRatio = fMean / (pfStds[i] + 0.000001);
		float fTilt = s_pAlignParam->GetTilt(i);
		printf(" %3d  %8.2f  %8.2f  %8.2f  %8.2f\n", i, fTilt, 
		   pfMeans[i], pfStds[i], fRatio);
	}
	printf("\n");
	//----------------------------------------
	int iZeroTilt = s_pAlignParam->GetFrameIdxFromTilt(0.0f);
	float fTol = s_fThreshold * (float)fabs(pfMeans[iZeroTilt]) 
	   / (pfStds[iZeroTilt] + 0.000001f);
	//-----------------------------------
	for(int i=0; i<iAllFrms; i++)
	{	float fRatio = (float)fabs(pfMeans[i]) / (pfStds[i] + 0.000001f);
		if(fRatio > fTol) continue;
		//-------------------------
		int iSecIdx = s_pAlignParam->GetSecIndex(i);
		float fTilt = s_pAlignParam->GetTilt(i);
		pDarkFrames->Add(i, iSecIdx, fTilt);
	}
	delete[] pfMeans; delete[] pfStds;
	if(pDarkFrames->m_iNumDarks <= 0) return;
	//---------------------------------------	
	for(int i=pDarkFrames->m_iNumDarks-1; i>=0; i--)
	{	int iFrmIdx = pDarkFrames->GetFrmIdx(i);
		float fTilt = pAlignParam->GetTilt(iFrmIdx);
		pTomoStack->RemoveFrame(iFrmIdx);
		pAlignParam->RemoveFrame(iFrmIdx);
		printf("Remove image at %.2f deg: \n", fTilt);
	}
}


CRemoveDarkFrames::CRemoveDarkFrames(void)
{
}

CRemoveDarkFrames::~CRemoveDarkFrames(void)
{
}

void CRemoveDarkFrames::Run
(	Util::CNextItem* pNextItem,
	int iGpuID,
	float* pfMeans,
	float* pfStds
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	m_pfMeans = pfMeans;
	m_pfStds = pfStds;
	this->Start();
}

void CRemoveDarkFrames::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	int iPixels = s_pTomoStack->GetPixels();
	size_t tBytes = sizeof(float) * iPixels;
	float *gfImg = 0L, *gfBuf = 0L;
	cudaMalloc(&gfImg, tBytes);
	cudaMalloc(&gfBuf, tBytes);
	//-------------------------
	Util::GCalcMoment2D calcMoment2D;
	bool bPadded = true;
	calcMoment2D.SetSize(s_pTomoStack->m_aiStkSize, !bPadded);
	//--------------------------------------------------------
	float afMeanStd[2] = {0.0f};
	while(true)
	{	int iFrame = m_pNextItem->GetNext();
		if(iFrame < 0) break;
		//-------------------
		float* pfFrame = s_pTomoStack->GetFrame(iFrame);
		cudaMemcpy(gfImg, pfFrame, tBytes, cudaMemcpyDefault);
		m_pfMeans[iFrame] = calcMoment2D.DoIt(gfImg, 1, true);
		m_pfStds[iFrame] = calcMoment2D.DoIt(gfImg, 2, true)
		   - m_pfMeans[iFrame] * m_pfMeans[iFrame];
		if(m_pfStds[iFrame] <= 0) m_pfStds[iFrame] = 0.0f;
		else m_pfStds[iFrame] = (float)sqrtf(m_pfStds[iFrame]);
	}
	if(gfImg != 0L) cudaFree(gfImg);
	if(gfBuf != 0L) cudaFree(gfBuf);
}
