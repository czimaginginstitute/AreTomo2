#include "CUtilInc.h"
#include <Util/Util_LinEqs.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>

using namespace Util;

CRemoveSpikes1D::CRemoveSpikes1D(void)
{
	m_iDim = 4;
	m_iWinSize = 11;
	m_pfTerms = new float[m_iDim];
	m_pfFit = new float[m_iDim];
	m_pfMatrix = new float[m_iDim * m_iDim];
}

CRemoveSpikes1D::~CRemoveSpikes1D(void)
{
	if(m_pfTerms != 0L) delete[] m_pfTerms;
	if(m_pfFit != 0L) delete[] m_pfFit;
	if(m_pfMatrix != 0L) delete[] m_pfMatrix;
}

void CRemoveSpikes1D::SetWinSize(int iWinSize)
{
	m_iWinSize = iWinSize;
}

void CRemoveSpikes1D::DoIt(float* pfData, int iSize)
{
	float* pfDataX = new float[iSize];
	for(int i=0; i<iSize; i++) pfDataX[i] = (float)i;
	this->DoIt(pfDataX, pfData, iSize);
	if(pfDataX != 0L) delete[] pfDataX;
}

void CRemoveSpikes1D::DoIt
(	float* pfDataX,
	float* pfDataY,
	int iSize
)
{	m_pfDataX = pfDataX;
	m_pfDataY = pfDataY;
	m_iSize = iSize;
	if(m_iWinSize > iSize) return;
	if(m_iWinSize < 5) return;
	//------------------------
	m_pfOldY = new float[m_iSize];
	m_pfWeight = new float[m_iSize];
	memcpy(m_pfOldY, m_pfDataY, sizeof(float) * m_iSize);
	memset(m_pfWeight, 0, sizeof(float) * m_iSize);
	//---------------------------------------------
	for(int i=0; i<iSize; i++)
	{	int iStart = i - m_iWinSize / 2;
		iStart = (iStart < 0) ? 0 : iStart;
		if((iStart + m_iWinSize) > m_iSize)
		{	iStart = m_iSize - m_iWinSize;
		}
		m_pfWeight[i] = 1.0f; 
	}
	//---------------------------
	for(int i=0; i<m_iSize; i++)
	{	int iStart = i - m_iWinSize / 2;
		iStart = (iStart < 0) ? 0 : iStart;
		if((iStart + m_iWinSize) > m_iSize) 
		{	iStart = m_iSize - m_iWinSize;
		}
		bool bSuccess = mQuadraticFit(iStart);
		if(!bSuccess) continue;
		//---------------------
		mCalcTerms(i, 1.0f);
		float fFitVal = 0.0f;
		for(int t=0; t<m_iDim; t++)
		{	fFitVal += (m_pfFit[t] * m_pfTerms[t]);
		}
		float fDiff = (float)fabs(m_pfDataY[i] - fFitVal) + 1.0f;
		m_pfWeight[i] = 1.0f / fDiff;
	}
	//----------------------------------
	for(int i=0; i<m_iSize; i++)
	{	int iStart = i - m_iWinSize / 2;
		iStart = (iStart < 0) ? 0 : iStart;
		if((iStart + m_iWinSize) > m_iSize)
		{	iStart = m_iSize - m_iWinSize;
		}
		bool bSuccess = mQuadraticFit(iStart);
		if(!bSuccess) continue;
		//---------------------
		mCalcTerms(i, 1.0f);
		float fFitVal = 0.0f;
                for(int t=0; t<m_iDim; t++)
		{	fFitVal += (m_pfFit[t] * m_pfTerms[t]);
		}
		if(fabs(m_pfDataY[i]) > fabs(fFitVal)) 
		{	m_pfDataY[i] = fFitVal;
		}
	}
	if(m_pfOldY != 0L) delete[] m_pfOldY;
	if(m_pfWeight != 0L) delete[] m_pfWeight;
	m_pfOldY = 0L;
	m_pfWeight = 0L;
}

bool CRemoveSpikes1D::mQuadraticFit(int iStart)
{	
	int iDimSqr = m_iDim * m_iDim;
	memset(m_pfMatrix, 0, sizeof(float) * iDimSqr);
	memset(m_pfFit, 0, sizeof(float) * m_iDim);
	//---------------------------------------
	for(int w=0; w<m_iWinSize; w++)
	{	int x = iStart + w;
		mCalcTerms(x, m_pfWeight[x]);
		for(int r=0; r<m_iDim; r++)
		{	for(int c=0; c<m_iDim; c++)
			{	int i = r * m_iDim + c;
				m_pfMatrix[i] += (m_pfTerms[r] 
				* m_pfTerms[c] / m_iWinSize);
			}
			m_pfFit[r] += (m_pfTerms[r] * m_pfOldY[x] 
			* m_pfWeight[x] / m_iWinSize);
		}
	}
	//--------------------------------------------------------------
	Util_LinEqs aLinEqs;
	bool bSuccess = aLinEqs.DoIt(m_pfMatrix, m_pfFit, m_iDim);
	return bSuccess;
}

void CRemoveSpikes1D::mCalcTerms(int x, float fW)
{
	m_pfTerms[0] = fW;
	for(int i=1; i<m_iDim; i++)
	{	m_pfTerms[i] = m_pfTerms[i-1] * m_pfDataX[x];
	}
}

