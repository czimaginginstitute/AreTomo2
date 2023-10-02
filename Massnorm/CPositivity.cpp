#include "CMassNormInc.h"
#include <memory.h>
#include <stdio.h>

using namespace MassNorm;

CPositivity::CPositivity(void)
{
	m_fMissingVal = (float)-1e10;
}

CPositivity::~CPositivity(void)
{
}

void CPositivity::DoIt(MrcUtil::CTomoStack* pTomoStack)
{
	m_pTomoStack = pTomoStack;
	//------------------------
	m_fMin = mCalcMin(0);
	for(int i=1; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	float fMin = mCalcMin(i);
		if(fMin < m_fMin) m_fMin = fMin;
	}
	if(m_fMin >= 0) return;
	//---------------------
	for(int i=0; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	printf("...... Set positivity %4d, %4d left\n",
			i+1, m_pTomoStack->m_aiStkSize[2]-1-i);
		mSetPositivity(i); 
	}
	printf("\n");
}

float CPositivity::mCalcMin(int iFrame)
{
	float fMin = 10000000.0f;
	float* pfFrame = m_pTomoStack->GetFrame(iFrame);
	int iPixels = m_pTomoStack->m_aiStkSize[0]
		* m_pTomoStack->m_aiStkSize[1];
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		if(fMin > pfFrame[i]) fMin = pfFrame[i];
	}
	return fMin;
}

void CPositivity::mSetPositivity(int iFrame)
{
	float* pfFrame = m_pTomoStack->GetFrame(iFrame);
	int iPixels = m_pTomoStack->m_aiStkSize[0]
		* m_pTomoStack->m_aiStkSize[1];
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		else pfFrame[i] -= m_fMin;
	}
}

