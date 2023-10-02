#include "CCommonLineInc.h"
#include "../Util/CUtilInc.h"
#include <CuUtilFFT/GFFT1D.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace CommonLine;

static CLineSet* s_pLineSet = 0L;

cufftComplex* CSumLines::DoIt(CLineSet* pLineSet)
{
	s_pLineSet = pLineSet;
	//--------------------
	int iNumGpus = s_pLineSet->m_iNumGpus;
	CSumLines* pThreads = new CSumLines[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(i);
	}
	//-------------------------
	bool bClean = true;
	pThreads[0].WaitForExit(-1.0f);
	cufftComplex* gAllSum = pThreads[0].GetSum(bClean);
	if(iNumGpus == 1)
	{	delete[] pThreads;
		return gAllSum;
	}
	//---------------------
	int iCmpSize = s_pLineSet->m_iCmpSize;
	size_t tBytes = sizeof(cufftComplex) * iCmpSize;
	cufftComplex* gCmpBuf = 0L;
	cudaMalloc(&gCmpBuf, tBytes);
	//---------------------------
	GFunctions aGFunctions;
	for(int i=1; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
		cufftComplex* gCmpSum = pThreads[i].GetSum(bClean);
		cudaMemcpy(gCmpBuf, gCmpSum, tBytes, cudaMemcpyDefault);
		cudaFree(gCmpSum);
		//----------------
		aGFunctions.Sum
		( gAllSum, gCmpBuf, 1.0f, 1.0f, gAllSum, iCmpSize
		);
	}
	//--------------------------------------------------------
	cudaFree(gCmpBuf);
	delete[] pThreads;
	return gAllSum;
}

CSumLines::CSumLines(void)
{
	m_gCmpSum = 0L;
}

CSumLines::~CSumLines(void)
{
	this->Clean();
}

void CSumLines::Clean(void)
{
	if(m_gCmpSum != 0L) cudaFree(m_gCmpSum);
	m_gCmpSum = 0L;
}

cufftComplex* CSumLines::GetSum(bool bClean)
{
	cufftComplex* gCmpSum = m_gCmpSum;
	if(bClean) m_gCmpSum = 0L;
	return gCmpSum;
}

void CSumLines::Run(int iThreadID)
{	
	m_iGpuID = s_pLineSet->GetGpuID(iThreadID);
	m_iCmpSize = s_pLineSet->m_iCmpSize;
	this->Start();
}

void CSumLines::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	if(m_gCmpSum != 0L) cudaFree(m_gCmpSum);
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMalloc(&m_gCmpSum, tBytes);
	cudaMemset(m_gCmpSum, 0, tBytes);
	//-------------------------------
	GFunctions aGFunctions;
	for(int i=0; i<s_pLineSet->m_iNumProjs; i++)
	{	int iGpuID = s_pLineSet->GetLineGpu(i);
		if(iGpuID != m_iGpuID) continue;
		//------------------------------
		cufftComplex* gCmpLine = s_pLineSet->GetLine(i);
		aGFunctions.Sum
		( m_gCmpSum, gCmpLine, 1.0f, 1.0f,
		  m_gCmpSum, m_iCmpSize
		);
	}
}

