#include "CCommonLineInc.h"
#include <CuUtilFFT/GFFT1D.h>
#include <memory.h>
#include <stdio.h>

using namespace CommonLine;

static __global__ void mGInterpolate
(	float* gfPadLine1,
	float* gfPadLine2,
	float fW1,
	float fW2,
	float* gfPadLine,
	int iSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	gfPadLine[i] = fW1 * gfPadLine1[i] + fW2 * gfPadLine2[i]; 
}

GPossibleLines::GPossibleLines(void)
{
	m_pfPadLines = 0L;
	m_pfTiltAxes = 0L;
	m_pfTiltAngles = 0L;
}

GPossibleLines::~GPossibleLines(void)
{
	this->Clean();
}

void GPossibleLines::Clean(void)
{
	if(m_pfPadLines != 0L) delete[] m_pfPadLines;
	if(m_pfTiltAxes != 0L) delete[] m_pfTiltAxes;
	if(m_pfTiltAngles != 0L) delete[] m_pfTiltAngles;
	m_pfPadLines = 0L;
	m_pfTiltAxes = 0L;
	m_pfTiltAngles = 0L;
}

void GPossibleLines::Setup
(	int iNumLines,      // number of lines per image
	int iNumImgs,       // number of tilt images
	int iLineSize,      // number of pixels per line 
	float* pfAngRange,  // range of tilt axis, min and max
	float* pfTiltAngles
)
{	this->Clean();
	//------------
	m_iNumLines = iNumLines;
	m_iNumImgs = iNumImgs;
	m_iLineSize = iLineSize;
	m_iPadSize = (m_iLineSize / 2 + 1) * 2;
	//-------------------------------------
	m_pfTiltAxes = new float[m_iNumLines];
	float fStep = (pfAngRange[1] - pfAngRange[1]) / (m_iNumLines - 1);
	for(int i=0; i<m_iNumLines; i++)
	{	m_pfTiltAxes[i] = pfAngRange[0] + i * fStep;
	}
	//--------------------------------------------------
	m_pfTiltAngles = new float[m_iNumImgs];
	memcpy(m_pfTiltAngles, pfTiltAngles, sizeof(float) * m_iNumImgs);
	//---------------------------------------------------------------
	size_t tSize = m_iPadSize * m_iNumLines;
	tSize = tSize * m_iNumImgs;
	m_pfPadLines = new float[tSize];
}

void GPossibleLines::SetLine(int iImage, int iLine, float* gfPadLine)
{
	float* pfPadLine = this->GetLine(iImage, iLine);
	size_t tBytes = sizeof(float) * m_iPadSize;
	cudaMemcpy(pfPadLine, gfPadLine, tBytes, cudaMemcpyDefault);
}

void GPossibleLines::GetLine(int iImage, int iLine, float* gfPadLine)
{
	float* pfPadLine = this->GetLine(iImage, iLine);
	size_t tBytes = sizeof(float) * m_iPadSize;
	cudaMemcpy(gfPadLine, pfPadLine, tBytes, cudaMemcpyDefault);
}

float* GPossibleLines::GetLine(int iImage, int iLine)
{
	size_t tPixels = m_iPadSize * m_iNumLines;
	size_t tOffset = iImage * tPixels + iLine * m_iPadSize;
	return m_pfPadLines + tOffset;
}

float GPossibleLines::GetTiltAxis(int iLine)
{
	return m_pfTiltAxes[iLine];
}

float GPossibleLines::GetTilt(int iImage)
{
	return m_pfTiltAngles[iImage];
}

void GPossibleLines::Interpolate
(	int iImage, 
	float fTiltAxis, 
	float* gfPadLine
)
{	if(fTiltAxis <= m_pfTiltAxes[0])
	{	this->GetLine(iImage, 0, gfPadLine);
		return;
	}
	//-------------
	if(fTiltAxis >= m_pfTiltAxes[m_iNumLines-1])
	{	this->GetLine(iImage, m_iNumLines-1, gfPadLine);
		return;
	}
	//-------------
	float fStep = m_pfTiltAxes[1] - m_pfTiltAxes[0];
	int iLine1 = (int)((fTiltAxis - m_pfTiltAxes[0]) / fStep);
	int iLine2 = iLine1 + 1;
	float fTiltAxis1 = this->GetTiltAxis(iLine1);
	float fTiltAxis2 = this->GetTiltAxis(iLine2);
	float fW1 = (fTiltAxis2 - fTiltAxis) / (fTiltAxis2 - fTiltAxis1);
	float fW2 = (fTiltAxis - fTiltAxis1) / (fTiltAxis2 - fTiltAxis1);
	//---------------------------------------------------------------
	size_t tBytes = m_iPadSize * sizeof(float);
	float *gfPadLine1 = 0L, *gfPadLine2 = 0L;
	cudaMalloc(&gfPadLine1, tBytes);
	cudaMalloc(&gfPadLine2, tBytes);
	this->GetLine(iImage, iLine1, gfPadLine1);
	this->GetLine(iImage, iLine2, gfPadLine2);
	//----------------------------------------
	dim3 aBlockDim(256, 1);
	dim3 aGridDim(m_iLineSize / aBlockDim.x + 1, 1);
	mGInterpolate<<<aGridDim, aBlockDim>>>
	( gfPadLine1, gfPadLine2, fW1, fW2, gfPadLine, m_iLineSize
	);
	//--------------------------------------------------------
	cudaFree(gfPadLine1);
	cudaFree(gfPadLine2);
}
