#include "CCTFCorInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace CTFCor;

static __global__ void mGCalcCTF2d
(	int iCmpX,
	int iCmpY,
	CCTFParam* gCTFParam,
	float* gfCTF
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float fX = (float)(i % iCmpX);
	float fY = (float)(i / iCmpX);
	if(fY >= iCmpY) return;
	//---------------------
	fX = fX / (2 * (iCmpX - 1));
	fY = fY / iCmpY;
	if(fY > 0.5f) fY -= 1.0f;
	float fFreq = sqrtf(fX * fX + fY * fY);
	float fAzimuth = atan2f(fY, fX);
	//------------------------------
	fX = gCTFParam->m_fDefocusMax + gCTFParam->m_fDefocusMin;
	fY = gCTFParam->m_fDefocusMax - gCTFParam->m_fDefocusMin;
	float fDf = cosf(2.0f * (fAzimuth - gCTFParam->m_fAstAzimuth));
	fDf = 0.5f * (fX + fY * fDf);
	//---------------------------
	fX = fFreq * fFreq;
	fY = gCTFParam->m_fWL * gCTFParam->m_fWL;
	float fPhaseShift = 3.141592654f * gCTFParam->m_fWL * fX
		* (fDf - 0.5f * fX * fY * gCTFParam->m_fCs)
		+ gCTFParam->m_fPhaseShiftAmp
		+ gCTFParam->m_fPhaseShiftExt;
	//------------------------------------
	gfCTF[i] = -sinf(fPhaseShift);
}


GCalcCTF2D::GCalcCTF2D(void)
{
	m_gfCTF = 0L;
	m_fPI = (float)(4.0 * atan(1.0));
}

GCalcCTF2D::~GCalcCTF2D(void)
{
	this->Clean();
}

void GCalcCTF2D::Clean(void)
{
	if(m_gfCTF != 0L) cudaFree(m_gfCTF);
	m_gfCTF = 0L;
}

void GCalcCTF2D::Setup
(	float fPixelSize, // Angstrom
	float fKv, // keV
	float fCs, // nm
	float fAmpContrast,
	float fPhaseShiftExt // radian
)
{	m_fPixelSize = fPixelSize;
	//------------------------
	float fWL = GCalcCTF2D::CalcWaveLength(fKv);
	m_aCTFParam.m_fWL = fWL / m_fPixelSize;
	m_aCTFParam.m_fCs = (float)(fCs * 1e7) / m_fPixelSize;
	//----------------------------------------------------
	m_aCTFParam.m_fPhaseShiftAmp = (float)atan(fAmpContrast
		/ sqrt(1.0 - fAmpContrast));
	m_aCTFParam.m_fPhaseShiftExt = fmodf(fPhaseShiftExt, m_fPI);
}

void GCalcCTF2D::SetSize(int iCmpX, int iCmpY)
{
	this->Clean();
	//------------
	m_aiCmpSize[0] = iCmpX;
	m_aiCmpSize[1] = iCmpY;
	//---------------------
	size_t tBytes = sizeof(float) * m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gfCTF, tBytes);
}

void GCalcCTF2D::DoIt
(	float fDefocusMin,   // A
	float fDefocusMax,   // A
	float fAstAzimuth    // radian
)
{	if(m_gfCTF == 0L) return;
	//-----------------------
	m_aCTFParam.m_fDefocusMin = fDefocusMin / m_fPixelSize;
	m_aCTFParam.m_fDefocusMax = fDefocusMax / m_fPixelSize;
	m_aCTFParam.m_fAstAzimuth = fAstAzimuth;
	//--------------------------------------
	CCTFParam* gCTFParam = 0L;
	size_t tBytes = sizeof(CCTFParam);
	cudaMalloc(&gCTFParam, tBytes);
	cudaMemcpy(gCTFParam, &m_aCTFParam, tBytes, cudaMemcpyDefault);
	//-------------------------------------------------------------
	int iSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	dim3 aBlockDim(1024, 1);
	dim3 aGridDim(iSize / aBlockDim.x + 1, 1);
	//----------------------------------------
	mGCalcCTF2d<<<aGridDim, aBlockDim>>>
	( m_aiCmpSize[0], m_aiCmpSize[1], gCTFParam, m_gfCTF
	);
	cudaFree(gCTFParam);
	//------------------
	float fDefocus = (m_aCTFParam.m_fDefocusMin
		+ m_aCTFParam.m_fDefocusMax) / 2.0f;
	m_fFreq0 = (float)sqrtf((3.141593f - m_aCTFParam.m_fPhaseShiftAmp)
		/ (3.141593f * m_aCTFParam.m_fWL * fDefocus));
}

float GCalcCTF2D::CalcWaveLength(float fKv)
{
	double dWl = 12.26 / sqrt(fKv * 1000 + 0.9784 * fKv * fKv);
        return (float)dWl;
}
