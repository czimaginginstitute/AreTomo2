#include "CStreAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>

using namespace StreAlign;

CStretchCC2D::CStretchCC2D(void)
{
	m_gfImg1 = 0L;
	m_pfCCs = 0L;
	m_pfTiltOffsets = 0L;
}

CStretchCC2D::~CStretchCC2D(void)
{
	this->Clean();
}

void CStretchCC2D::Clean(void)
{
	if(m_gfImg1 != 0L) 
	{	cudaFree(m_gfImg1);
		m_gfImg1 = 0L;
	}
	if(m_pfCCs != 0L) 
	{	delete[] m_pfCCs;
		m_pfCCs = 0L;
	}
	if(m_pfTiltOffsets != 0L)
	{	delete[] m_pfTiltOffsets;
		m_pfTiltOffsets = 0L;
	}
}

void CStretchCC2D::SetImgSize(int* piImgSize, bool bPadded)
{
	this->Clean();
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iImageX = m_aiImgSize[0];
	if(m_bPadded) m_iImageX = (m_aiImgSize[0] / 2 - 1) * 2;
	//-----------------------------------------------------
	m_aiCCSize[0] = m_iImageX * 6 / 20 * 2;
	m_aiCCSize[1] = m_aiImgSize[1] * 8 / 20 * 2;
	//------------------------------------------
	int iImgSize = m_aiImgSize[0] * m_aiImgSize[1];
	int iCCSize = m_aiCCSize[0] * m_aiCCSize[1];
	size_t tBytes = (iImgSize * 3 + iCCSize * 2) * sizeof(float);
	//-----------------------------------------------------------
	cudaMalloc(&m_gfImg1, tBytes);
	m_gfImg2 = m_gfImg1 + iImgSize;
	m_gfStreImg = m_gfImg2 + iImgSize;
	//--------------------------------
	m_gfCCImg1 = m_gfStreImg + iImgSize;
	m_gfCCImg2 = m_gfCCImg1 + iCCSize;
}
	
void CStretchCC2D::DoIt
(	float* pfImg1,
	float* pfImg2,
	float fTilt1,
	float fTilt2,
	float* pfTiltOffsets,
	int iNumOffsets
)
{	size_t tBytes = sizeof(float) * m_aiImgSize[0] * m_aiImgSize[1];
	cudaMemcpy(m_gfImg1, pfImg1, tBytes, cudaMemcpyDefault);
	cudaMemcpy(m_gfImg2, pfImg2, tBytes, cudaMemcpyDefault);
	//------------------------------------------------------
	m_afTilt[0] = fTilt1;
	m_afTilt[1] = fTilt2;
	//-------------------
	if(m_pfTiltOffsets != 0L) delete[] m_pfTiltOffsets;
	m_pfTiltOffsets = new float[iNumOffsets];
	if(m_pfCCs != 0L) delete[] m_pfCCs;
	m_pfCCs = new float[iNumOffsets];
	m_iNumOffsets = iNumOffsets;
	//--------------------------
	for(int i=0; i<m_iNumOffsets; i++)
	{	mDoIt(i);
	}
}

void CStretchCC2D::mDoIt(int iOffset)
{
	double dRad = 4 * atan(1.0) / 180.0;
	double dCos1 = cos((m_afTilt[0] + m_pfTiltOffsets[iOffset]) * dRad);
	double dCos2 = cos((m_afTilt[1] + m_pfTiltOffsets[iOffset]) * dRad);
	//------------------------------------------------------------------
	bool bPadded = (m_aiImgSize[0] != m_iImageX) ? true : false;
	float* gfRefImg = 0L;
	Util::GStretch aGStretch;
	//-----------------------
	if(dCos1 < dCos2)
	{	float fSFact = (float)(dCos2 / dCos1);
		aGStretch.DoIt(m_gfImg1, m_aiImgSize, bPadded, 
			fSFact, 0.0f, m_gfStreImg);
		gfRefImg = m_gfImg2;
	}
	else
	{	float fSFact = (float)(dCos1 / dCos2);
		aGStretch.DoIt(m_gfImg2, m_aiImgSize, bPadded,
			fSFact, 0.0f, m_gfStreImg);
		gfRefImg = m_gfImg1;
	}
	//--------------------------
	mPartialCopy(gfRefImg, m_gfCCImg1);
	mPartialCopy(m_gfStreImg, m_gfCCImg2);
	//------------------------------------
	Util::GRealCC2D aGRealCC2D;
	float* gfBuf = m_gfStreImg;
	m_pfCCs[iOffset] = aGRealCC2D.DoIt
	( m_gfCCImg1, m_gfCCImg2, gfBuf, m_aiCCSize
	);
}

