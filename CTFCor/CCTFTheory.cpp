#include "CFindCTFInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace GCTFFind;

float CCTFParam::GetWavelength(bool bAngstrom)
{
	if(bAngstrom) return (m_fWavelength * m_fPixelSize);
	else return m_fWavelength;
}

float CCTFParam::GetDefocusMax(bool bAngstrom)
{
	if(bAngstrom) return (m_fDefocusMax * m_fPixelSize);
	else return m_fDefocusMax;
}

float CCTFParam::GetDefocusMin(bool bAngstrom)
{
	if(bAngstrom) return (m_fDefocusMin * m_fPixelSize);
	else return m_fDefocusMin;
}

CCTFTheory::CCTFTheory(void)
{
	m_fPI = (float)(4.0 * atan(1.0));
	m_pCTFParam = new CCTFParam;
	memset(m_pCTFParam, 0, sizeof(CCTFParam));
}

CCTFTheory::~CCTFTheory(void)
{
	if(m_pCTFParam != 0L) delete m_pCTFParam;
}

void CCTFTheory::Setup
(	float fKv, // keV
	float fCs, // mm
	float fAmpContrast,
	float fPixelSize,    // A
	float fAstTol,       // A, negative means no tolerance
	float fAddPhaseShift // radian
)
{	m_pCTFParam->m_fWavelength = mCalcWavelength(fKv) / fPixelSize;
	m_pCTFParam->m_fCs = (float)(fCs * 1e7 / fPixelSize);
	m_pCTFParam->m_fAmpContrast = fAmpContrast;
	m_pCTFParam->m_fAmpPhaseShift
	= atan(fAmpContrast / sqrt(1 - fAmpContrast));
	m_pCTFParam->m_fPixelSize = fPixelSize;
	m_pCTFParam->m_fAddPhaseShift = fmodf(fAddPhaseShift, m_fPI);
	m_fAstTol = fAstTol / m_pCTFParam->m_fPixelSize;
}

void CCTFTheory::SetDefocus
(	float fDefocusMin,   // A
	float fDefocusMax,   // A
	float fAstAzimuth    // Rad
)
{	m_pCTFParam->m_fDefocusMin = fDefocusMin / m_pCTFParam->m_fPixelSize;
	m_pCTFParam->m_fDefocusMax = fDefocusMax / m_pCTFParam->m_fPixelSize;
	m_pCTFParam->m_fAstAzimuth = fAstAzimuth;
}

void CCTFTheory::SetDefocusInPixel
(	float fDefocusMin, // pixel
	float fDefocusMax, // pixel
	float fAstAzimuth  // Radian
)
{	m_pCTFParam->m_fDefocusMin = fDefocusMin;
	m_pCTFParam->m_fDefocusMax = fDefocusMax;
	m_pCTFParam->m_fAstAzimuth = fAstAzimuth;
}

void CCTFTheory::SetParam
(	CCTFParam* pCTFParam
)
{	if(m_pCTFParam == pCTFParam) return;
	memcpy(m_pCTFParam, pCTFParam, sizeof(CCTFParam));
}

CCTFParam* CCTFTheory::GetParam(void)
{
	return m_pCTFParam;
}

float CCTFTheory::Evaluate
(	float fFreq,   // relative frquency in [-0.5, 0.5]
	float fAzimuth
)
{	float fPhaseShift = CalcPhaseShift(fFreq, fAzimuth);
	return (float)(-sin(fPhaseShift));
}

//------------------------------------------------------------------------------
// 1. Return number of extrema before the given spatial frequency.
//    Eq. 11 of Rohou & Grigoriff 2015
// 2. fFrequency is relative frequency in [-0.5, +0.5].
//------------------------------------------------------------------------------
int CCTFTheory::CalcNumExtrema
(	float fFreq,
	float fAzimuth
)
{	float fPhaseShift = CalcPhaseShift(fFreq, fAzimuth);
	int iNumExtrema = (int)(fPhaseShift / m_fPI + 0.5f);
	return iNumExtrema;
}

//-----------------------------------------------------------------------------
// 1. Return the spatial frequency of Nth zero.
//    The returned frequency is in 1/pixel.
//-----------------------------------------------------------------------------
float CCTFTheory::CalcNthZero(int iNthZero, float fAzimuth)
{
	float fPhaseShift = iNthZero * m_fPI;
	float fFreq = CalcFrequency(fPhaseShift, fAzimuth);
	return fFreq;
}

//-----------------------------------------------------------------------------
//  1. Calculate defocus in pixel at the given azumuth angle.
//  2. fDefocusMin, fDefocusMax are the min, max defocus at the major and
//     minor axis.
//  3. fAstAzimuth is the angle of the astimatism, or the major axis.
//  4. fDefocusMax must be larger than fDefocusMin.
//-----------------------------------------------------------------------------
float CCTFTheory::CalcDefocus(float fAzimuth)
{
	float fSumDf = m_pCTFParam->m_fDefocusMax
		+ m_pCTFParam->m_fDefocusMin;
	float fDifDf = m_pCTFParam->m_fDefocusMax
		- m_pCTFParam->m_fDefocusMin;
	double dCosA = cos(2.0 * (fAzimuth - m_pCTFParam->m_fAstAzimuth));
	float fDefocus = (float)(0.5 * (fSumDf + fDifDf * dCosA));
	return fDefocus;
}

float CCTFTheory::CalcPhaseShift
(	float fFreq,
	float fAzimuth
)
{	float fS2 = fFreq * fFreq;
	float fW2 = m_pCTFParam->m_fWavelength
		* m_pCTFParam->m_fWavelength;
	float fDefocus = CalcDefocus(fAzimuth);
	float fPhaseShift = m_fPI * m_pCTFParam->m_fWavelength * fS2
		* (fDefocus - 0.5f * fW2 * fS2 * m_pCTFParam->m_fCs)
		+ m_pCTFParam->m_fAmpPhaseShift
		+ m_pCTFParam->m_fAddPhaseShift;
	return fPhaseShift;
}

//-----------------------------------------------------------------------------
// 1. Returen spatial frequency in 1/pixel given phase shift and fAzimuth
//    in radian.
//-----------------------------------------------------------------------------
float CCTFTheory::CalcFrequency
(	float fPhaseShift,
	float fAzimuth
)
{	float fDefocus = CalcDefocus(fAzimuth);
	double dW3 = pow(m_pCTFParam->m_fWavelength, 3.0);
	double a = -0.5 * m_fPI * dW3 * m_pCTFParam->m_fCs;
	double b = m_fPI * m_pCTFParam->m_fWavelength * fDefocus;
	double c = m_pCTFParam->m_fAddPhaseShift
		+ m_pCTFParam->m_fAmpPhaseShift;
	double dDet = b * b - 4.0 * a * (c - fPhaseShift);
	//------------------------------------------------
	if(m_pCTFParam->m_fCs == 0)
	{	double dFreq2 = (fPhaseShift - c) / b;
		if(dFreq2 > 0) return (float)sqrt(dFreq2);
		else return 0.0f;
	}
	else if(dDet < 0.0)
	{	return 0.0f;
	}
	else
	{	double dSln1 = (-b + sqrt(dDet)) / (2 * a);
		double dSln2 = (-b - sqrt(dDet)) / (2 * a);
		if(dSln1 > 0) return (float)sqrt(dSln1);
		else if(dSln2 > 0) return (float)sqrt(dSln2);
		else return 0.0f;
	}
}

//-----------------------------------------------------------------------------
// 1. Compare if this instance is almost equal to the argument pCTF in terms
//    of their member values.
// 2. fDfTol is the tolerance of defocus comparison.
// 3. Other tolerances are hard-coded.
//-----------------------------------------------------------------------------
bool CCTFTheory::EqualTo(CCTFTheory* pCTF, float fDfTol)
{
	CCTFParam* pParam = pCTF->GetParam();
	float fDif = m_pCTFParam->m_fDefocusMax - pParam->m_fDefocusMax;
	if(fabs(fDif) > fDfTol) return false;
	//-----------------------------------
	fDif = m_pCTFParam->m_fDefocusMin - pParam->m_fDefocusMin;
	if(fabs(fDif) > fDfTol) return false;
	//-----------------------------------
	fDif = m_pCTFParam->m_fCs - pParam->m_fCs;
	if(fabs(fDif) > 0.01) return false;
	//---------------------------------
	fDif = m_pCTFParam->m_fWavelength - pParam->m_fWavelength;
	if(fabs(fDif) > 1e-4) return false;
	//---------------------------------
	fDif = m_pCTFParam->m_fAmpContrast - pParam->m_fAmpContrast;
	if(fabs(fDif) > 1e-4) return false;
	//---------------------------------
	float f5Degree = 0.0277f;
	double dDif = fabs(m_pCTFParam->m_fAddPhaseShift
		- pParam->m_fAddPhaseShift);
	dDif = fmod(fDif, 2.0 * m_fPI);
	if(dDif > f5Degree) return false;
	//-------------------------------
	dDif = fabs(m_pCTFParam->m_fAstAzimuth - pParam->m_fAstAzimuth);
	dDif = fmod(dDif, m_fPI);
	if(dDif > f5Degree) return false;
	//-------------------------------
	return true;
}

CCTFTheory* CCTFTheory::GetCopy(void)
{
	CCTFTheory* pCopy = new CCTFTheory;
	pCopy->SetParam(this->GetParam());
	pCopy->m_fAstTol = m_fAstTol;
	return pCopy;
}

float CCTFTheory::GetPixelSize(void)
{
     return m_pCTFParam->m_fPixelSize;
}

//-----------------------------------------------------------------------------
// Given acceleration voltage in keV, return the electron wavelength.
//-----------------------------------------------------------------------------
float CCTFTheory::mCalcWavelength(float fKv)
{
	double dWl = 12.26 / sqrt(fKv * 1000 + 0.9784 * fKv * fKv);
	return (float)dWl;
}

//-----------------------------------------------------------------------------
// Enforce that m_fDefocusMax > m_fDefocusMin and -90 < m_fAstAzimuth < 90.
//-----------------------------------------------------------------------------
void CCTFTheory::mEnforce(void)
{
	m_pCTFParam->m_fAstAzimuth -= m_fPI
		* round(m_pCTFParam->m_fAstAzimuth / m_fPI);
	if(m_pCTFParam->m_fDefocusMax < m_pCTFParam->m_fDefocusMin)
	{	float fTemp = m_pCTFParam->m_fDefocusMax;
		m_pCTFParam->m_fDefocusMax = m_pCTFParam->m_fDefocusMin;
		m_pCTFParam->m_fDefocusMin = fTemp;
	}
}
