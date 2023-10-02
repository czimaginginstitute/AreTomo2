#include "CMrcUtilInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace MrcUtil;

static float* s_pfProj = 0L;
static int* s_piProjSize = 0L;
static int s_aiPatchSize[2] = {0};
static int s_aiNumPatches[2] = {7, 7};
static int* s_piCenters = 0L;
static float* s_pfStds = 0L;

void CFindObjectCenter::DoIt
(	CTomoStack* pTomoStack,
	CAlignParam* pAlignParam,
	int* piGpuIDs,
	int iNumGpus,
	float* pfShift
)
{	int iZeroTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
	s_pfProj = pTomoStack->GetFrame(iZeroTilt);
	s_piProjSize = pTomoStack->m_aiStkSize;
	s_aiPatchSize[0] = s_piProjSize[0] * 3 / 4;
	s_aiPatchSize[1] = s_piProjSize[1] * 3 / 4;
	//-------------------------------------
	int iNumPatches = s_aiNumPatches[0] * s_aiNumPatches[1];
	s_piCenters = new int[2 * iNumPatches];
	for(int i=0; i<iNumPatches; i++)
	{	int j = 2 * i;
		int x = i % s_aiNumPatches[0] - s_aiNumPatches[0] / 2;
		int y = i / s_aiNumPatches[1] - s_aiNumPatches[1] / 2;
		s_piCenters[j] = x * 70 + s_piProjSize[0] / 2;
		s_piCenters[j+1] = y * 70 + s_piProjSize[1] / 2; 
	}
	s_pfStds = new float[iNumPatches];
	//--------------------------------
	Util::CNextItem nextItem;
	nextItem.Create(iNumPatches);
	//---------------------------
	CFindObjectCenter* pThreads = new CFindObjectCenter[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	//----------------
	int iMax = 0;
	float fMaxStd = s_pfStds[0];
	for(int i=0; i<iNumPatches; i++)
	{	int j =  2 * i;
		printf("...... %2d  %4d  %4d  sigma %.2f\n", i+1,
			s_piCenters[j], s_piCenters[j+1], s_pfStds[i]);
		if(fMaxStd >= s_pfStds[i]) continue;
		iMax = i;
		fMaxStd = s_pfStds[i];
	}
	pfShift[0] = s_piCenters[iMax * 2] - s_piProjSize[0] / 2;
	pfShift[1] = s_piCenters[iMax * 2 + 1] - s_piProjSize[1] / 2;
	printf("New center: %8.2f  %8.2f\n\n", pfShift[0], pfShift[1]);
	//-------------------------------------------------------------
	delete[] s_piCenters;
	delete[] s_pfStds;
}


CFindObjectCenter::CFindObjectCenter(void)
{
}

CFindObjectCenter::~CFindObjectCenter(void)
{
}

void CFindObjectCenter::Run
(	Util::CNextItem* pNextItem,
	int iGpuID
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}

void CFindObjectCenter::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	int iPixels = s_aiPatchSize[0] * s_aiPatchSize[1];
	size_t tBytes = sizeof(float) * iPixels;
	float* gfPatch = 0L;
	cudaMalloc(&gfPatch, tBytes);
	//---------------------------
	Util::GCalcMoment2D calcMoment2D;
	bool bPadded = true;
	tBytes = sizeof(float) * s_aiPatchSize[0];
	//----------------------------------------
	while(true)
	{	int i = m_pNextItem->GetNext();
		if(i < 0) break;
		int iStartX = s_piCenters[2 * i] + s_aiPatchSize[0] / 2;
		int iStartY = s_piCenters[2 * i + 1] + s_aiPatchSize[1] / 2;
		int iOffset = iStartY * s_piProjSize[0] + iStartX;
		for(int y=0; y<s_aiPatchSize[1]; y++)
		{	float* gfDst = gfPatch + y * s_aiPatchSize[0];
			float* pfSrc = s_pfProj + y * s_piProjSize[0] + iOffset;
			cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault); 
		}
		//----------------------------------------------------------
		calcMoment2D.Setup(1, 0.0);
		float fMean = calcMoment2D.DoIt(gfPatch, s_aiPatchSize, !bPadded);
		calcMoment2D.Setup(2, 0.0f);
		float fMean2 = calcMoment2D.DoIt(gfPatch, s_aiPatchSize, !bPadded);
		fMean2 = fMean2 - fMean * fMean;
		if(fMean2 <= 0) s_pfStds[i] = 0.0f;
		else s_pfStds[i] = (float)sqrt(fMean2);
	}
	if(gfPatch != 0L) cudaFree(gfPatch);
}
