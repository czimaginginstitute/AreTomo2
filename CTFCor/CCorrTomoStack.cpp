#include "CCTFCorInc.h"
#include <CuUtilFFT/GFFT2D.h>
#include <memory.h>
#include <stdio.h>

using namespace CTFCor;

static MrcUtil::CTomoStack* s_pTomoStack = 0L;
static float s_fKv = 0.0f;
static float s_fCs = 0.0f;
static float s_fPixelSize = 0.0f;
static float s_fDefocus = 0.0f;

void CCorrTomoStack::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	float fKv,
	float fCs,
	float fPixelSize,
	float fDefocus,
	int* piGpuIDs,
	int iNumGpus
)
{	s_pTomoStack = pTomoStack;
	s_fKv = fKv;
	s_fCs = fCs;
	s_fPixelSize = fPixelSize;
	s_fDefocus = fDefocus;
	//--------------------
	bool bCTF = fKv > 0 && fCs > 0 && fPixelSize > 0 && fDefocus > 0;
	if(bCTF) printf("Start CTF correction and Fourier cropping...\n");
	else return; //printf("Start Fourier cropping...\n");
	Util::CNextItem nextItem;
	nextItem.Create(pTomoStack->m_aiStkSize[2]);
	//------------------------------------------
	CCorrTomoStack* pThreads = new CCorrTomoStack[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(&nextItem, piGpuIDs[i]);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	if(bCTF) printf("CTF correction and Fourier cropping: done.\n\n");
	else printf("Fourier cropping: done.\n\n");
}

CCorrTomoStack::CCorrTomoStack(void)
{
	m_gfCTF = 0L;
	m_gCmpImg = 0L;
}

CCorrTomoStack::~CCorrTomoStack(void)
{
	this->Clean();
}

void CCorrTomoStack::Clean(void)
{
	if(m_gfCTF != 0L) cudaFree(m_gfCTF);
	if(m_gCmpImg != 0L) cudaFree(m_gCmpImg);
	m_gfCTF = 0L;
	m_gCmpImg = 0L;	
}

void CCorrTomoStack::Run
(	Util::CNextItem* pNextItem,
	int iGpuID
)
{	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}

void CCorrTomoStack::ThreadMain(void)
{
	this->Clean();
	cudaSetDevice(m_iGpuID);
	//----------------------
	m_aiCmpSize[0] = s_pTomoStack->m_aiStkSize[0] / 2 + 1;
	m_aiCmpSize[1] = s_pTomoStack->m_aiStkSize[1];
	//--------------------------------------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gCmpImg, sizeof(cufftComplex) * iCmpSize);
	mCalcCTF();
	//---------
	bool bPad = true;
	m_aForwardFFT.CreateForwardPlan(s_pTomoStack->m_aiStkSize, !bPad);
	m_aInverseFFT.CreateInversePlan(s_pTomoStack->m_aiStkSize, !bPad);	
	//----------------------------------------------------------------
	while(true)
	{	int iProj = m_pNextItem->GetNext();
		if(iProj < 0) break;
		mCorrectProj(iProj);
		if(m_gfCTF == 0L) continue;
	}
	//-------------------------------------------------------------
	m_aForwardFFT.DestroyPlan();
	m_aInverseFFT.DestroyPlan();
	m_aGCorCTF2D.Clean();
	this->Clean();
}

void CCorrTomoStack::mCalcCTF(void)
{
	if(s_fPixelSize <= 0) return;
	else if(s_fKv <= 0) return;
	else if(s_fCs <= 0) return;
	//-------------------------
	GCalcCTF2D aGCalcCTF2D;
	aGCalcCTF2D.Setup(s_fPixelSize, s_fKv, s_fCs, 0.07f, 0.0f);
	aGCalcCTF2D.SetSize(m_aiCmpSize[0], m_aiCmpSize[1]);
	aGCalcCTF2D.DoIt(s_fDefocus, s_fDefocus, 0.0f);
	//---------------------------------------------
	m_gfCTF = aGCalcCTF2D.m_gfCTF;
	aGCalcCTF2D.m_gfCTF = 0L;
	m_fFreq0 = aGCalcCTF2D.m_fFreq0;
	printf("First zero: %f\n", m_fFreq0);
	//-----------------------------------
	m_aGCorCTF2D.SetSize(m_aiCmpSize[0], m_aiCmpSize[1]);

	/*
	Util::CSaveTempMrc saveTempMrc;
	saveTempMrc.OpenFile("/home/szheng/SambaFolder/Temp/TestCtf", ".mrc");
	saveTempMrc.SetSize(m_aiCmpSize, 1);
	saveTempMrc.GDoIt(m_gfCTF, 0);
	*/
}

void CCorrTomoStack::mCorrectProj(int iProj)
{
	mForwardFFT(iProj);
	if(m_gfCTF != 0L)
	{	m_aGCorCTF2D.DoIt(m_gCmpImg, m_gfCTF, m_fFreq0);
	}
	m_aGCorBilinear.DoIt(m_gCmpImg, m_aiCmpSize);
	mInverseFFT(iProj);
}

void CCorrTomoStack::mForwardFFT(int iProj)
{
	float* pfProj = s_pTomoStack->GetFrame(iProj);
	Util::CPad2D pad2D;
	pad2D.Pad(pfProj, s_pTomoStack->m_aiStkSize, (float*)m_gCmpImg);
	//--------------------------------------------------------------
	bool bNorm = true;
	m_aForwardFFT.Forward((float*)m_gCmpImg, bNorm);
}

void CCorrTomoStack::mInverseFFT(int iProj)
{
	m_aInverseFFT.Inverse(m_gCmpImg);
	//-------------------------------
	float* pfProj = s_pTomoStack->GetFrame(iProj);
	Util::CPad2D pad2D;
	int aiPadSize[] = {2 * m_aiCmpSize[0], m_aiCmpSize[1]};
	pad2D.Unpad((float*)m_gCmpImg, aiPadSize, pfProj);
}
	
