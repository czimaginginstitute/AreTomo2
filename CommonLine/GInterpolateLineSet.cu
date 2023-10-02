#include "CCommonLineInc.h"
#include "../Util/CUtilInc.h"
#include <CuUtilFFT/GFFT1D.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace CommonLine;

static __global__ void mGInterpolate
(	cufftComplex* gCmpLine1,
	cufftComplex* gCmpLine2,
	int iCmpSize,
	float fWeight,
	cufftComplex* gCmpResult
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iCmpSize) return;
	gCmpResult[i].x = fWeight * gCmpLine1[i].x
		+ (1 - fWeight) * gCmpLine2[i].x;
	gCmpResult[i].y = fWeight * gCmpLine1[i].y
		+ (1 - fWeight) * gCmpLine2[i].y;
}

static CPossibleLines* s_pPossibleLines = 0L;
static float* s_pfRotAngles = 0L;
static CLineSet* s_pLineSet = 0L;

void GInterpolateLineSet::DoIt
(	CPossibleLines* pPossibleLines,
	float* pfRotAngles,
	CLineSet* pLineSet
)
{	s_pPossibleLines = pPossibleLines;
	s_pfRotAngles = pfRotAngles;
	s_pLineSet = pLineSet;
	//--------------------
	int iNumGpus = s_pLineSet->m_iNumGpus;
	GInterpolateLineSet* pThreads = new GInterpolateLineSet[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(i);
	}
	//-------------------------
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
}

GInterpolateLineSet::GInterpolateLineSet(void)
{
	m_gCmpLine1 = 0L;
	m_gCmpLine2 = 0L;
}

GInterpolateLineSet::~GInterpolateLineSet(void)
{
	this->Clean();
}

void GInterpolateLineSet::Clean(void)
{
	if(m_gCmpLine1 != 0L) cudaFree(m_gCmpLine1);
	if(m_gCmpLine2 != 0L) cudaFree(m_gCmpLine2);
	m_gCmpLine1 = 0L;
	m_gCmpLine2 = 0L;
}

void GInterpolateLineSet::Run(int iThreadID)
{	
	m_iGpuID = s_pLineSet->GetGpuID(iThreadID);
	m_iCmpSize = s_pLineSet->m_iCmpSize;
	this->Start();
}

void GInterpolateLineSet::ThreadMain(void)
{
	cudaSetDevice(m_iGpuID);
	//----------------------
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMalloc(&m_gCmpLine1, tBytes);
	cudaMalloc(&m_gCmpLine2, tBytes);
	//-------------------------------
	GFunctions aGFunctions;
	for(int i=0; i<s_pLineSet->m_iNumProjs; i++)
	{	int iGpuID = s_pLineSet->GetLineGpu(i);
		if(iGpuID != m_iGpuID) continue;
		mInterpolate(i);
	}
}

void GInterpolateLineSet::mInterpolate(int iProj)
{
	int iNumLines = s_pPossibleLines->m_iNumLines;
	float fRotAngle = s_pfRotAngles[iProj];
	//-------------------------------------
	float fLine = s_pPossibleLines->CalcLinePos(fRotAngle);
	int iLine1 = (int)fLine;
	if(iLine1 < 0) iLine1 = 0;
	//------------------------
	int iLine2 = iLine1 + 1;
	if(iLine2 >= iNumLines) 
	{	iLine2 = iNumLines - 1;
		iLine1 = iLine2 - 1;
	}
	//--------------------------
	float fW = 1.0f - (fLine - iLine1);
	if(iLine1 == 0) fW = 1.0f;
	else if(iLine2 == (iNumLines - 1)) fW = 0.0f;
	//-------------------------------------------
	s_pPossibleLines->GetLine(iProj, iLine1, m_gCmpLine1);
	s_pPossibleLines->GetLine(iProj, iLine2, m_gCmpLine2);
	//----------------------------------------------------
	cufftComplex* gCmpRes = s_pLineSet->GetLine(iProj);
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(m_iCmpSize / aBlockDim.x + 1, 1);
	mGInterpolate<<<aGridDim, aBlockDim>>>
	( m_gCmpLine1, m_gCmpLine2, m_iCmpSize,
	  fW, gCmpRes
	);
}
