#include "CDoseWeightInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace DoseWeight;

static MrcUtil::CTomoStack* s_pTomoStack = 0L;
static bool s_bDoseWeight = false;

void CWeightTomoStack::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	int* piGpuIDs,
	int iNumGpus
)
{	s_pTomoStack = pTomoStack;
	s_bDoseWeight = s_pTomoStack->bHasAcqIndices() &&
	   s_pTomoStack->m_fImgDose > 0;
	if(s_bDoseWeight)
	{	printf("Dose weighting and Fourier cropping ......\n");
	}
	else printf("Start Fourier cropping ......\n");
	//-----------------
	Util::CNextItem nextItem;
	nextItem.Create(pTomoStack->m_aiStkSize[2]);
	//-----------------
	CWeightTomoStack* pThreads = new CWeightTomoStack[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	//-----------------
	if(s_bDoseWeight) 
	{	printf("Dose weighting and Fourier cropping of "
		   "tilt series: done.\n\n");
	}
	else
	{	printf("Fourier cropping of tilt series: done\n\n");
	}
}

CWeightTomoStack::CWeightTomoStack(void)
{
	m_pGDoseWeightImg = 0L;
	m_gCmpImg = 0L;
	m_pfDoses = 0L;
}

CWeightTomoStack::~CWeightTomoStack(void)
{
	this->Clean();
}

void CWeightTomoStack::Clean(void)
{
	if(m_pGDoseWeightImg != 0L) delete m_pGDoseWeightImg;
	if(m_gCmpImg != 0L) cudaFree(m_gCmpImg);
	if(m_pfDoses != 0L) delete[] m_pfDoses;
	m_pGDoseWeightImg = 0L;
	m_gCmpImg = 0L;
	m_pfDoses = 0L;
}

void CWeightTomoStack::Run
(	Util::CNextItem* pNextItem,
	int iGpuID
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}

void CWeightTomoStack::ThreadMain(void)
{
	this->Clean();
	cudaSetDevice(m_iGpuID);
	//-----------------
	m_aiCmpSize[0] = s_pTomoStack->m_aiStkSize[0] / 2 + 1;
	m_aiCmpSize[1] = s_pTomoStack->m_aiStkSize[1];
	//-----------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gCmpImg, sizeof(cufftComplex) * iCmpSize);
	//-----------------
	m_aForwardFFT.CreatePlan(s_pTomoStack->m_aiStkSize, true);
	m_aInverseFFT.CreatePlan(s_pTomoStack->m_aiStkSize, false);	
	//-----------------
	if(s_bDoseWeight) mSetupDoseWeight();
	//-----------------
	while(true)
	{	int iProj = m_pNextItem->GetNext();
		if(iProj < 0) break;
		mCorrectProj(iProj);
		printf("  image %4d has been processed.\n", iProj);
	}
	//-----------------
	m_aForwardFFT.DestroyPlan();
	m_aInverseFFT.DestroyPlan();
	this->Clean();
}

void CWeightTomoStack::mSetupDoseWeight(void)
{	
	float fImgDose = s_pTomoStack->m_fImgDose;
	m_pfDoses = new float[s_pTomoStack->m_aiStkSize[2]];
	for(int i=0; i<s_pTomoStack->m_aiStkSize[2]; i++)
	{	int iAcqIdx = s_pTomoStack->m_piAcqIndices[i];
		m_pfDoses[i] = (iAcqIdx + 1) * fImgDose;
	}
	//-----------------
	CInput* pInput = CInput::GetInstance();
	m_pGDoseWeightImg = new GDoseWeightImage;
	m_pGDoseWeightImg->BuildWeight(pInput->m_fPixelSize, 
	   pInput->m_fKv, m_pfDoses,
	   s_pTomoStack->m_aiStkSize);
}

void CWeightTomoStack::mCorrectProj(int iProj)
{
	mForwardFFT(iProj);
	mDoseWeight(iProj);
	mInverseFFT(iProj);
}

void CWeightTomoStack::mForwardFFT(int iProj)
{
	float* pfProj = s_pTomoStack->GetFrame(iProj);
	Util::CPad2D pad2D;
	pad2D.Pad(pfProj, s_pTomoStack->m_aiStkSize, (float*)m_gCmpImg);
	//--------------------------------------------------------------
	bool bNorm = true;
	m_aForwardFFT.Forward((float*)m_gCmpImg, bNorm);
}

void CWeightTomoStack::mDoseWeight(int iProj)
{
	if(m_pGDoseWeightImg == 0L) return;
	float fDose = m_pfDoses[iProj];
	m_pGDoseWeightImg->DoIt(m_gCmpImg, fDose);
}

void CWeightTomoStack::mInverseFFT(int iProj)
{
	m_aInverseFFT.Inverse(m_gCmpImg);
	//-------------------------------
	Util::CPad2D aPad2D;
	float* pfProj = s_pTomoStack->GetFrame(iProj);
	int aiPadSize[] = {2 * m_aiCmpSize[0], m_aiCmpSize[1]};
	aPad2D.Unpad((float*)m_gCmpImg, aiPadSize, pfProj);
}
	
