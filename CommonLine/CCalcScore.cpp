#include "CCommonLineInc.h"
#include <CuUtilFFT/GFFT1D.h>
#include <Util/Util_LinEqs.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace CommonLine;

static CLineSet* s_pLineSet = 0L;
static cufftComplex* s_gCmpSum = 0L;

float CCalcScore::DoIt
(	CLineSet* pLineSet,
	cufftComplex* gCmpSum
)
{	s_pLineSet = pLineSet;
	s_gCmpSum = gCmpSum;
	//------------------
	int iNumGpus = s_pLineSet->m_iNumGpus;
	CCalcScore* pThreads = new CCalcScore[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(i);
	}
	//-------------------------
	float fCCSum = 0.0f;
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
		fCCSum += pThreads[i].m_fCCSum;
	}
	delete[] pThreads;
	//----------------
	float fCC = fCCSum / s_pLineSet->m_iNumProjs;
	return fCC;
}

CCalcScore::CCalcScore(void)
{
	m_gCmpSum = 0L;
	m_gCmpRef = 0L;
}

CCalcScore::~CCalcScore(void)
{
}

void CCalcScore::Run(int iThreadID)
{
	m_iThreadID = iThreadID;
	m_iGpuID = s_pLineSet->GetGpuID(m_iThreadID);
	m_iCmpSize = s_pLineSet->m_iCmpSize;
	this->Start();
}

void CCalcScore::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	m_gCmpSum = s_gCmpSum;
	if(m_iThreadID != 0)
	{	m_gCmpSum = mCudaMallocLine(false);
		size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
		cudaMemcpy(m_gCmpSum, s_gCmpSum, tBytes, cudaMemcpyDefault);
	}
	m_gCmpRef = mCudaMallocLine(false);
	//---------------------------------
	m_fCCSum = 0.0f;
	for(int i=0; i<s_pLineSet->m_iNumProjs; i++)
	{	int iGpuID = s_pLineSet->GetLineGpu(i);
		if(iGpuID != m_iGpuID) continue;
		//------------------------------
		float fCC = mCorrelate(i);
		m_fCCSum += fCC;
	}
	//----------------------
	if(m_gCmpRef != 0L) cudaFree(m_gCmpRef);
	if(m_iThreadID != 0) cudaFree(m_gCmpSum);
	m_gCmpRef = 0L;
	m_gCmpSum = 0L;
}

float CCalcScore::mCorrelate(int iLine)
{
	cufftComplex* gCmpLine = s_pLineSet->GetLine(iLine);
	//--------------------------------------------------
	GFunctions aGFunctions;
	aGFunctions.Sum
	( m_gCmpSum, gCmpLine, 1.0f, -1.0f,
	  m_gCmpRef, m_iCmpSize
	);
	//---------------------
	Util::GCC1D aGCC1D;
	aGCC1D.SetBFactor(10);
	float fCC = aGCC1D.DoIt(m_gCmpRef, gCmpLine, m_iCmpSize);
	return fCC;
}

cufftComplex* CCalcScore::mCudaMallocLine(bool bZero)
{
	cufftComplex* gCmpLine = 0L;
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMalloc(&gCmpLine, tBytes);
	if(bZero) cudaMemset(gCmpLine, 0, tBytes);
	return gCmpLine;
}
