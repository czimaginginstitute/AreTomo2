#include "CCommonLineInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace CommonLine;

CLineSet::CLineSet(void)
{
	m_pgCmpLines = 0L;
	m_piLineGpus = 0L;
	m_piGpuIDs = 0L;
	m_iNumProjs = 0;
	m_iCmpSize = 0;
}

CLineSet::~CLineSet(void)
{
	this->Clean();
}
/*
void CLineSet::Clean(void)
{
	if(m_piLineGpus != 0L) delete[] m_piLineGpus;
	m_piLineGpus = 0L;
	//----------------
	if(m_pgCmpLines == 0L) return;
	for(int i=0; i<m_iNumProjs; i++)
	{	cufftComplex* gCmpLine = m_pgCmpLines[i];
		if(gCmpLine == 0L) continue;
		else cudaFree(gCmpLine);
	}
	delete m_pgCmpLines;
	m_pgCmpLines = 0L;
}
*/

void CLineSet::Clean(void)
{
	if(m_pgCmpLines == 0L) return;
	//----------------------------
	int iCurGpu = -1;
	cudaGetDevice(&iCurGpu);
	//----------------------
	int iGpuID = -1;
	for(int i=0; i<m_iNumProjs; i++)
	{	if(m_pgCmpLines[i] == 0L) continue;
		//---------------------------------
		if(iGpuID != m_piLineGpus[i])
		{	iGpuID = m_piLineGpus[i];
			cudaSetDevice(iGpuID);
		}
		//----------------------------
		cudaFree(m_pgCmpLines[i]);
	}
	delete[] m_pgCmpLines; m_pgCmpLines = 0L;
	delete[] m_piLineGpus; m_piLineGpus = 0L;
	//---------------------------------------
	if(iCurGpu >= 0) cudaSetDevice(iCurGpu);
}

void CLineSet::Setup(void)
{
	int iCurGpu = -1;
	cudaGetDevice(&iCurGpu);
	//----------------------	
	this->Clean();	
	//------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance();
	m_iNumProjs = pClParam->m_pTomoStack->m_aiStkSize[2];
	m_iCmpSize = pClParam->m_iCmpLineSize;
	//------------------------------------
	CInput* pInput = CInput::GetInstance();
	m_iNumGpus = pInput->m_iNumGpus;
	m_piGpuIDs = pInput->m_piGpuIDs;
	//------------------------------
	m_piLineGpus = new int[m_iNumProjs];
	//----------------------------------
	Util::CSplitItems splitItems;
	splitItems.Create(m_iNumProjs, m_iNumGpus);
	//-----------------------------------------
	m_pgCmpLines = new cufftComplex*[m_iNumProjs];
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	for(int i=0; i<m_iNumGpus; i++)
	{	int iGpuID = m_piGpuIDs[i];
		cudaSetDevice(iGpuID);
		//--------------------
		int iStart = splitItems.GetStart(i);
		int iSize = splitItems.GetSize(i);
		cufftComplex* gCmpLine = 0L;
		for(int j=0; j<iSize; j++)
		{	int iLine = iStart + j;
			m_piLineGpus[iLine] = iGpuID;
			//---------------------------
			cudaMalloc(&gCmpLine, tBytes);
			m_pgCmpLines[iLine] = gCmpLine;
		}
	}
	//--------------------------------------------
	if(iCurGpu >= 0) cudaSetDevice(iCurGpu);
}

cufftComplex* CLineSet::GetLine(int iLine)
{	
	cufftComplex* gCmpLine = m_pgCmpLines[iLine];
	return gCmpLine;
}

int CLineSet::GetLineGpu(int iLine)
{
	return m_piLineGpus[iLine];
}

int CLineSet::GetGpuID(int iNthGpu)
{
	return m_piGpuIDs[iNthGpu];
}
