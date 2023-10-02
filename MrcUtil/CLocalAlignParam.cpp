#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace MrcUtil;

CLocalAlignParam::CLocalAlignParam(void)
{
	m_pfCoordXs = 0L;
	m_iNumParams = 5;
}

CLocalAlignParam::~CLocalAlignParam(void)
{
	this->Clean();
}

void CLocalAlignParam::Clean(void)
{
	if(m_pfCoordXs == 0L) return;
	cudaFreeHost(m_pfCoordXs);
	m_pfCoordXs = 0L;
}

void CLocalAlignParam::Setup(int iNumTilts, int iNumPatches)
{
	this->Clean();
	m_iNumTilts = iNumTilts;
	m_iNumPatches = iNumPatches;
	//--------------------------
	int iSize = m_iNumTilts * m_iNumPatches;
	int iBytes = sizeof(float) * iSize * m_iNumParams;
	cudaMallocHost(&m_pfCoordXs, iBytes);
	m_pfCoordYs = m_pfCoordXs + iSize;
	m_pfShiftXs = m_pfCoordXs + iSize * 2;
	m_pfShiftYs = m_pfCoordXs + iSize * 3;
	m_pfGoodShifts = m_pfCoordXs + iSize * 4;
	//--------------------------------------
	memset(m_pfCoordXs, 0, iBytes);
}

void CLocalAlignParam::GetParam(int iTilt, float* gfAlnParam)
{
	int iSize = m_iNumTilts * m_iNumPatches;
	int iOffset = iTilt * m_iNumPatches;
	int iBytes = m_iNumPatches * sizeof(float);
	for(int i=0; i<m_iNumParams; i++)
	{	float* pfSrc = m_pfCoordXs + i * iSize + iOffset;
		float* gfDst = gfAlnParam + i * m_iNumPatches;
		cudaMemcpy(gfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

void CLocalAlignParam::GetCoordXYs
(	int iTilt, 
	float* pfCoordXs, 
	float* pfCoordYs
)
{	int iBytes = m_iNumPatches * sizeof(float);
	int iOffset = iTilt * m_iNumPatches;
	memcpy(pfCoordXs, m_pfCoordXs + iOffset, iBytes);
	memcpy(pfCoordYs, m_pfCoordYs + iOffset, iBytes); 
}

void CLocalAlignParam::SetCoordXY(int iTilt, int iPatch, float fX, float fY)
{
	int i = iTilt * m_iNumPatches + iPatch;
	m_pfCoordXs[i] = fX;
	m_pfCoordYs[i] = fY;
}

void CLocalAlignParam::SetShift(int iTilt, int iPatch, float fSx, float fSy)
{
	int i = iTilt * m_iNumPatches + iPatch;
	m_pfShiftXs[i] = fSx;
	m_pfShiftYs[i] = fSy;
}

void CLocalAlignParam::SetBad(int iTilt, int iPatch, bool bBad)
{
	int i = iTilt * m_iNumPatches + iPatch;
	if(bBad) m_pfGoodShifts[i] = 0.0f;
	else m_pfGoodShifts[i] = 1.0f;
}

