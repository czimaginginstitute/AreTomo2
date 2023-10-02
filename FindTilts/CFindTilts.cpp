#include "CFindTiltsInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace FindTilts;

CFindTilts::CFindTilts(void)
{
	m_pfTilts = 0L;
	m_pfTiltAxes = 0L;
}

CFindTilts::~CFindTilts(void)
{
	this->Clean();
}

void CFindTilts::Clean(void)
{
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_pfTiltAxes != 0L) delete[] m_pfTiltAxes;
	m_pfTilts = 0L;
	m_pfTiltAxes = 0L;	
}

void CFindTilts::Find
(	MrcUtil::CTomoStack* pTomoStack,
	float* pfTiltRange,
	int* piGpuIDs,
	int iNumGpus
)
{	this->Clean();
	//------------
	m_pTomoStack = pTomoStack;
	memcpy(m_afTiltRange, pfTiltRange, sizeof(float) * 2);
	m_piGpuIDs = piGpuIDs;
	m_iNumGpus = iNumGpus;
	//--------------------
	printf("Find tilt angles of projections.\n");
	m_iNumProjs = m_pTomoStack->m_aiProjSize[2];
	m_pfTilts = new float[m_iNumProjs];
	m_pfTiltAxes = new float[m_iNumProjs];
	for(int i=0; i<m_iNumProjs; i++)
	{	m_pfTiltAxes[i] = m_pTomoStack->GetTiltAxis(i);
	}
	//-----------------------------------------------------
	mFindTilts();
	this->Clean();	
}

void CFindTilts::mFindTilts(void)
{
	int iBestStart = -1;
	int iBestEnd = -1;
	float fMaxCC = -1000.0f;
	//----------------------
	int iSubProjs = m_iNumProjs / 4;
	int iEndProj = m_iNumProjs - 1;
	for(int i=0; i<iSubProjs; i++)
	{	float fCC = mCalcCC(i, iEndProj);
		printf("...... CC: %4d  %.5e\n", i, fCC);
		if(fCC < fMaxCC) continue;
		iBestStart = i;
		fMaxCC = fCC;
	}
	//-------------------
	for(int i=1; i<iSubProjs; i++)
	{	iEndProj = m_iNumProjs - 1 - i;
		float fCC = mCalcCC(iBestStart, iEndProj);
		printf("...... CC: %4d  %.5e\n", iEndProj, fCC);
		if(fCC < fMaxCC) continue;
		iBestEnd = iEndProj;
		fMaxCC = fCC;
	}
	//--------------------------
	mCalcTilts(iBestStart, iBestEnd);
	printf("Tilt angle distribution:\n");
	for(int i=0; i<m_iNumProjs; i++)
	{	m_pTomoStack->SetTiltA(i, m_pfTilts[i]);
		printf("...... %4d  %.2f\n", i+1, m_pfTilts[i]);
	}
	printf("\n");
}

float CFindTilts::mCalcCC(int iStartProj, int iEndProj)
{
	int* piProjSize = m_pTomoStack->m_aiProjSize;
	mCalcTilts(iStartProj, iEndProj);
	//-------------------------------
	float fCC = CTotalCC::DoIt
	( m_pTomoStack->m_ppfProjs, m_pfTilts, m_pfTiltAxes,
	  piProjSize, m_piGpuIDs, m_iNumGpus
	);
	return fCC;
}

void CFindTilts::mCalcTilts(int iStartProj, int iEndProj)
{
	float fRange = m_afTiltRange[1] - m_afTiltRange[0];
	float fStep = fRange / (iEndProj - iStartProj); 
        for(int i=0; i<m_iNumProjs; i++)
        {       if(i< iStartProj) m_pfTilts[i] = m_afTiltRange[0];
                else if(i > iEndProj) m_pfTilts[i] = m_afTiltRange[1];
                else m_pfTilts[i] = m_afTiltRange[0] + fStep * (i - iStartProj);
        }
}
