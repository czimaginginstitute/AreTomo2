#include "CCommonLineInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include <CuUtilFFT/GFFT1D.h>
#include <memory.h>
#include <stdio.h>

using namespace CommonLine;

void mFindMMM(float* gfPadLine, int iPadSize)
{
	int iSize = (iPadSize / 2 - 1) * 2;
	float* pfLine = new float[iSize];
	cudaMemcpy(pfLine, gfPadLine, iSize * sizeof(float), cudaMemcpyDefault);
	float fMin = pfLine[0];
	float fMax = pfLine[0];
	float fMean = 0;
	for(int i=0; i<iSize; i++)
	{	if(pfLine[i] < fMin) fMin = pfLine[i];
		else if(pfLine[i] > fMax) fMax = pfLine[i];
		if(pfLine[i] > 0) fMean += (pfLine[i] / iSize);
	}
	delete[] pfLine;
}

void mFindImageMMM(float* gfImg, int* piImgSize)
{
	int iPixels = piImgSize[0] * piImgSize[1];
	size_t tBytes = sizeof(float) * iPixels;
	float* pfImg = new float[iPixels];
	cudaMemcpy(pfImg, gfImg, tBytes, cudaMemcpyDefault);
	//--------------------------------------------------
	float fMin = pfImg[0];
	float fMax = pfImg[1];
	float fMean = 0.0f;
	for(int i=0; i<iPixels; i++)
	{	if(pfImg[i] < fMin) fMin = pfImg[i];
		else if(pfImg[i] > fMax) fMax = pfImg[i];
		if(pfImg[i] > 0) fMean += (pfImg[i] / iPixels);
	}
	delete[] pfImg;
}

static CPossibleLines* s_pPossibleLines = 0L;

CPossibleLines* CGenLines::DoIt(void)
{	
	s_pPossibleLines = new CPossibleLines;
	s_pPossibleLines->Setup();
	//------------------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance();
	int iNumFrames = pClParam->m_pTomoStack->m_aiStkSize[2];
	Util::CNextItem nextItem;
	nextItem.Create(iNumFrames);
	//--------------------------
	CInput* pInput = CInput::GetInstance();
	CGenLines* pThreads = new CGenLines[pInput->m_iNumGpus];
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	pThreads[i].Run(i, &nextItem);
	}
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
	//----------------
	CPossibleLines* pPossibleLines = s_pPossibleLines;
	s_pPossibleLines = 0L;
	return pPossibleLines;
}

CGenLines::CGenLines(void)
{
	m_gCmpPlane = 0L;
}

CGenLines::~CGenLines(void)
{
}

void CGenLines::mClean(void)
{
	if(m_gCmpPlane != 0L) cudaFree(m_gCmpPlane);
	m_gCmpPlane = 0L;
	//---------------
	m_calcComRegion.Clean();
	m_genComLine.Clean();
	m_fft1D.DestroyPlan();
}

void CGenLines::Run
(	int iThreadID,
	Util::CNextItem* pNextItem
)
{	CInput* pInput = CInput::GetInstance();
	m_iGpuID = pInput->m_piGpuIDs[iThreadID];
	m_pNextItem = pNextItem;
	this->Start();
}

void CGenLines::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance();
	m_iNumLines = pClParam->m_iNumLines;
	m_iLineSize = pClParam->m_iLineSize;
	//----------------------------------
	m_calcComRegion.DoIt();
	mGenLines();
	//----------
	this->mClean();
}

void CGenLines::mGenLines(void)
{
	if(m_gCmpPlane != 0L) cudaFree(m_gCmpPlane);
	int iCmpLineSize = m_iLineSize / 2 + 1;
	size_t tBytes = sizeof(cufftComplex) * m_iNumLines * iCmpLineSize;
	cudaMalloc(&m_gCmpPlane, tBytes);
	//-------------------------------
	bool bForward = true;
	m_fft1D.CreatePlan(m_iLineSize, m_iNumLines, bForward);
	//-----------------------------------------------------
	bool bPadded = true;
	m_genComLine.Setup();
	//-------------------
	while(true)
	{	int iProj = m_pNextItem->GetNext();
		if(iProj < 0) break;
		mGenProjLines(iProj);
	}
}

void CGenLines::mGenProjLines(int iProj)
{
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance();
	MrcUtil::CTomoStack* pTomoStack = pClParam->m_pTomoStack;
	MrcUtil::CAlignParam* pAlignParam = pClParam->m_pAlignParam;
	//----------------------------------------------------------
	float* pfProj = pTomoStack->GetFrame(iProj);
	float fTiltAngle = pAlignParam->GetTilt(iProj);
	float afShift[2] = {0.0f};
	pAlignParam->GetShift(iProj, afShift);
	//------------------------------------
	int* giComRegion = m_calcComRegion.m_giComRegion;
	m_genComLine.DoIt(iProj, giComRegion, (float*)m_gCmpPlane);
	//---------------------------------------------------------
	mForwardFFT();
	s_pPossibleLines->SetPlane(iProj, m_gCmpPlane);
}

void CGenLines::mForwardFFT(void)
{
	int iPadSize = (m_iLineSize / 2 + 1) * 2;
	float* gfPadLines = (float*)m_gCmpPlane;
	GRemoveMean removeMean;
	for(int i=0; i<m_iNumLines; i++)
	{	float* gfPadLine = gfPadLines + i * iPadSize;
		removeMean.DoIt(gfPadLine, iPadSize);
	}
	//-------------------------------------------
	bool bNorm = true;
	m_fft1D.Forward(gfPadLines, !bNorm);
}
