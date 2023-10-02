#include "CMassNormInc.h"
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace MassNorm;

CLinearNorm::CLinearNorm(void)
{
	m_fMissingVal = (float)-1e20;
}

CLinearNorm::~CLinearNorm(void)
{
}

void CLinearNorm::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	printf("Linear mass normalization...\n");
	m_pTomoStack = pTomoStack;
	//------------------------
	m_aiStart[0] = m_pTomoStack->m_aiStkSize[0] * 1 / 6;
	m_aiStart[1] = m_pTomoStack->m_aiStkSize[1] * 1 / 6;
	m_aiSize[0] = m_pTomoStack->m_aiStkSize[0] * 4 / 6;
	m_aiSize[1] = m_pTomoStack->m_aiStkSize[1] * 4 / 6;
	m_iNumFrames = m_pTomoStack->m_aiStkSize[2];
	//------------------------------------------
	float* pfMean = new float[m_iNumFrames];
	for(int i=0; i<m_iNumFrames; i++)
	{	pfMean[i] = mCalcMean(i);	
	}
	//-------------------------------
	int iZeroTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
	m_fRefMean = pfMean[iZeroTilt];
	if(m_fRefMean > 1000.0f) m_fRefMean = 1000.0f;
	for(int i=0; i<m_iNumFrames; i++)
	{	float fScale = m_fRefMean / (pfMean[i] + 0.00001f);
		mScale(i, fScale);
	}
	if(pfMean != 0L) delete[] pfMean;
	printf("Linear mass normalization: done.\n\n");
}

void CLinearNorm::mSmooth(float* pfMeans)
{
	double dMean = 0, dStd = 0;
	for(int i=0; i<m_iNumFrames; i++)
	{	dMean += pfMeans[i];
		dStd += (pfMeans[i] * pfMeans[i]);
	}
	dMean /= m_iNumFrames;
	dStd = dStd / m_iNumFrames - dMean * dMean;
	if(dStd <= 0) return;
	else dStd = sqrtf(dStd);
	//----------------------
	float fMin = (float)(dMean - 3 * dStd);	
	if(fMin <= 0) return;
	//-------------------
	for(int i=0; i<m_iNumFrames; i++)
	{	if(pfMeans[i] > fMin) continue;
		else pfMeans[i] = (float)dMean;
	}
}

void CLinearNorm::FlipInt(MrcUtil::CTomoStack* pTomoStack)
{
	m_pTomoStack = pTomoStack;
	m_iNumFrames = m_pTomoStack->m_aiStkSize[2];
	for(int i=0; i<m_iNumFrames; i++)
	{	mFlipInt(i);
	}
}

float CLinearNorm::mCalcMean(int iFrame)
{
	double dMean = 0.0;
	int iCount = 0;
	float* pfFrame = m_pTomoStack->GetFrame(iFrame);
	int iOffset = m_aiStart[1] * m_pTomoStack->m_aiStkSize[0]
		+ m_aiStart[0];
	for(int y=0; y<m_aiSize[1]; y++)
	{	int i = y * m_pTomoStack->m_aiStkSize[0] + iOffset;
		for(int x=0; x<m_aiSize[0]; x++)
		{	float fVal = pfFrame[i+x];
			if(fVal <= m_fMissingVal) continue;
			dMean += fVal;
			iCount++;
		}
	}
	if(iCount == 0) return 0.0f;
	dMean = dMean / iCount;
	return (float)dMean;
}

void CLinearNorm::mScale(int iFrame, float fScale)
{
	float* pfFrame = m_pTomoStack->GetFrame(iFrame);
	int iPixels = m_pTomoStack->m_aiStkSize[0]
		* m_pTomoStack->m_aiStkSize[1];
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		pfFrame[i] *= fScale;
	}
}

void CLinearNorm::mFlipInt(int iFrame)
{
	float* pfFrame = m_pTomoStack->GetFrame(iFrame);
	int iPixels = m_pTomoStack->m_aiStkSize[0]
		* m_pTomoStack->m_aiStkSize[1];
	float fMin = (float)1e20;
	float fMax = (float)-1e20;
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		if(pfFrame[i] < fMin) fMin = pfFrame[i];
		else if(pfFrame[i] > fMax) fMax = pfFrame[i];
		pfFrame[i] *= (-1);
	}
	//-------------------------
	float fOffset = fMax + fMin;
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		pfFrame[i] += fOffset;
	}
}
