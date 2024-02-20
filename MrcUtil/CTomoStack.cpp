#include "CMrcUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace MrcUtil;

CTomoStack::CTomoStack(void)
{
	m_ppfFrames = 0L;
	m_ppfCenters = 0L;
	m_fPixSize = 1.0f;
	m_fImgDose = 0.0f;
	//-----------------
	m_piAcqIndices = 0L;
	m_piSecIndices = 0L;
	m_pfTilts = 0L;
}

CTomoStack::~CTomoStack(void)
{
	this->Clean();
}

void CTomoStack::Clean(void)
{
	mCleanFrames();
	mCleanCenters();
	//-----------------
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_piAcqIndices != 0L) delete[] m_piAcqIndices;
	m_pfTilts = 0L;
	m_piAcqIndices = 0L;
	m_piSecIndices = 0L;
}

int CTomoStack::GetPixels(void)
{
	return m_aiStkSize[0] * m_aiStkSize[1];
}

int CTomoStack::GetNumFrames(void)
{
	return m_aiStkSize[2];
}

void CTomoStack::Create(int* piStkSize)
{
	this->Clean();
	//-----------------
	memcpy(m_aiStkSize, piStkSize, sizeof(int) * 3);
	m_ppfFrames = new float*[m_aiStkSize[2]];
	memset(m_ppfFrames, 0, sizeof(float*) * m_aiStkSize[2]);	
	//-----------------
	m_ppfCenters = new float*[m_aiStkSize[2]];
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	float* pfCent = new float[2];
		pfCent[0] = 0.5f * m_aiStkSize[0];
		pfCent[1] = 0.5f * m_aiStkSize[1];
		m_ppfCenters[i] = pfCent;
	} 
	//-----------------
	int iPixels = this->GetPixels();
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	m_ppfFrames[i] = new float[iPixels];
	}
	//-----------------
	m_pfTilts = new float[m_aiStkSize[2]];
	memset(m_pfTilts, 0, sizeof(float) * m_aiStkSize[2]);
	//-----------------
	int iSize = m_aiStkSize[2] * 2;
	m_piAcqIndices = new int[iSize];
	m_piSecIndices = &m_piAcqIndices[m_aiStkSize[2]];
	memset(m_piAcqIndices, 0, sizeof(float) * iSize);
}

void CTomoStack::SetFrame(int iFrame, float* pfFrame)
{
	int iPixels = this->GetPixels();
	size_t tBytes = sizeof(float) * iPixels;
	float* pfDst = m_ppfFrames[iFrame];
	if(pfDst == 0L) pfDst = new float[iPixels];
	//-----------------------------------------
	cudaMemcpy(pfDst, pfFrame, tBytes, cudaMemcpyDefault);
	m_ppfFrames[iFrame] = pfDst;
}

float* CTomoStack::GetFrame(int iProj)
{
	if(iProj < 0 || iProj >= m_aiStkSize[2]) return 0L;
	return m_ppfFrames[iProj];
}

void CTomoStack::GetFrame(int iFrame, float* pfFrame)
{
	float* pfSrc = this->GetFrame(iFrame);
	if(pfSrc == 0L) return;
	//---------------------
	int iPixels = this->GetPixels();
	size_t tBytes = sizeof(float) * iPixels;
	cudaMemcpy(pfFrame, pfSrc, tBytes, cudaMemcpyDefault);
}

void CTomoStack::SetCenter(int iProj, float* pfCent)
{
	float* pfDstCent = m_ppfCenters[iProj];	
	pfDstCent[0] = pfCent[0];
	pfDstCent[1] = pfCent[1];
}

void CTomoStack::GetCenter(int iProj, float* pfCent)
{
	float* pfSrcCent = m_ppfCenters[iProj];
	pfCent[0] = pfSrcCent[0];
	pfCent[1] = pfSrcCent[1];
}

void CTomoStack::SetTilts(float* pfTilts)
{
	memcpy(m_pfTilts, pfTilts, sizeof(float) * m_aiStkSize[2]);
}

void CTomoStack::SetAcqs(int* piAcqIndices)
{
	memcpy(m_piAcqIndices, piAcqIndices, sizeof(int) * m_aiStkSize[2]);
}

void CTomoStack::SetSecs(int* piSecIndices)
{
	memcpy(m_piSecIndices, piSecIndices, sizeof(int) * m_aiStkSize[2]);
}

void CTomoStack::SortByTilt(void)
{
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	float fMinT = m_pfTilts[i];
		int iMin = i;
		for(int j=i+1; j<m_aiStkSize[2]; j++)
		{	if(m_pfTilts[j] >= fMinT) continue;
			fMinT = m_pfTilts[j];
			iMin = j;
		}
		mSwap(i, iMin);
	}
}

void CTomoStack::SortByAcq(void)
{
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	int iMinAcq = m_piAcqIndices[i];
		int iMin = i;
		for(int j=i+1; j<m_aiStkSize[2]; j++)
		{	if(m_piAcqIndices[j] >= iMinAcq) continue;
			iMinAcq = m_piAcqIndices[j];
			iMin = j;
		}
		mSwap(i, iMin);
	}
}

//-----------------------------------------------------------------------------
// 1. This is usually called when a tilt series is sorted by tilt angle and
//    will be saved into a MRC file.
// 2. This function will keep the in-memory and on-disk tilt series have the
//    same section indices.
//-----------------------------------------------------------------------------
void CTomoStack::ResetSecIndices(void)
{
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	m_piSecIndices[i] = i;
	}
}

bool CTomoStack::bHasTiltAngles(void)
{
	if(m_aiStkSize[2] <= 0) return false;
	//-----------------
	int iMid = m_aiStkSize[2] / 2;
	float fDif = m_pfTilts[iMid] - m_pfTilts[0];
	if(fabs(fDif) <= 1.0f) return false;
	else return true;
}

//-----------------------------------------------------------------------------
// 1. When Acq indices are not given in the angle file (using -AngFile), dose
//    weighting will not be performed, and
// 2. ordered list file (.csv) will not be generated even -OutImod 1 is used.
//-----------------------------------------------------------------------------
bool CTomoStack::bHasAcqIndices(void)
{
	if(m_aiStkSize[2] <= 0) return false;
	//-----------------
	int iMid = m_aiStkSize[2] / 2;
	int iLast = m_aiStkSize[2] - 1;
	if(m_piAcqIndices[0] == m_piAcqIndices[iLast]) return false;
	if(m_piAcqIndices[0] == m_piAcqIndices[iMid]) return false;
	return true;
}

CTomoStack* CTomoStack::GetCopy(void)
{
	CTomoStack* pCopy = new CTomoStack;
	pCopy->Create(m_aiStkSize);
	//-----------------
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	pCopy->SetFrame(i, m_ppfFrames[i]);
		pCopy->SetCenter(i, m_ppfCenters[i]);
	}
	//-----------------
	pCopy->SetTilts(m_pfTilts);
	pCopy->SetAcqs(m_piAcqIndices);
	pCopy->SetSecs(m_piSecIndices);
	//-----------------
	return pCopy;
}

//-------------------------------------------------------------------
// 1. Get portion of the current tilt series. The portion means both
//    a portion of projection and subset of projections.
// 2. piStart: an array of three elements. piStart[0] and piStart[1]
//    are the starting x and y coordinates of and piStart[2] is the 
//    starting projection index.
// 3. piSize: an array of three elements. piSize[0] and piSize[1]
//    are the x and y sizes. piSize[2] is the number of projections.
//-------------------------------------------------------------------
CTomoStack* CTomoStack::GetSubStack(int* piStart, int* piSize)
{
	CTomoStack* pSubStack = new CTomoStack;
	pSubStack->Create(piSize);
	//-----------------
	int iBytes = sizeof(float) * piSize[0];
	int iOffset = piStart[1] * m_aiStkSize[0] + piStart[0];
	//-----------------
	float afCent[2] = {0.0f};
	afCent[0] = piStart[0] + 0.5f * piSize[0];
	afCent[1] = piStart[1] + 0.5f * piSize[1];
	//-----------------
	for (int i=0; i<piSize[2]; i++)
	{	int iSrcProj = i + piStart[2];
		float* pfSrcFrm = m_ppfFrames[iSrcProj] + iOffset;
		float* pfDstFrm = pSubStack->GetFrame(i);
		for(int y=0; y<piSize[1]; y++)
		{	float* pfSrc = pfSrcFrm + y * m_aiStkSize[0];
			float* pfDst = pfDstFrm + y * piSize[0];
			memcpy(pfDst, pfSrc, iBytes);
		}
		pSubStack->SetCenter(i, afCent);
	}
	//-----------------
	pSubStack->SetTilts(m_pfTilts);
	pSubStack->SetAcqs(m_piAcqIndices);
	pSubStack->SetSecs(m_piSecIndices);
	//-----------------
	return pSubStack;			
}

void CTomoStack::RemoveFrame(int iFrame)
{
	if(m_ppfFrames[iFrame] != 0L) delete[] m_ppfFrames[iFrame];
	if(m_ppfCenters[iFrame] != 0L) delete[] m_ppfCenters[iFrame];
	//-----------------
	for(int i=iFrame+1; i<m_aiStkSize[2]; i++)
	{	int j = i - 1;
		m_ppfFrames[j] = m_ppfFrames[i];
		m_ppfCenters[j] = m_ppfCenters[i];
		//----------------
		m_pfTilts[j] = m_pfTilts[i];
		m_piAcqIndices[j] = m_piAcqIndices[i];
		m_piSecIndices[j] = m_piSecIndices[i];
	};
	//-----------------
	int iLast = m_aiStkSize[2] - 1;
	m_ppfFrames[iLast] = 0L;
	m_ppfCenters[iLast] = 0L;
	m_aiStkSize[2] = iLast;
}

void CTomoStack::GetAlignedFrameSize(float fTiltAxis, int* piAlnSize)
{
	memcpy(piAlnSize, m_aiStkSize, sizeof(int) * 2);
	double dRot = fabs(sin(fTiltAxis * 3.14 / 180));
	if(dRot <= 0.707) return;
	piAlnSize[0] = m_aiStkSize[1];
	piAlnSize[1] = m_aiStkSize[0];
}

void CTomoStack::mCleanFrames(void)
{
	if(m_ppfFrames == 0L) return;
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	if(m_ppfFrames[i] == 0L) continue;
		delete[] m_ppfFrames[i];
	}
	delete[] m_ppfFrames;
	m_ppfFrames = 0L;
}

void CTomoStack::mCleanCenters(void)
{
	if(m_ppfCenters == 0L) return;
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	if(m_ppfCenters[i] == 0L) continue;
		delete[] m_ppfCenters[i];
	}
	delete[] m_ppfCenters;
	m_ppfCenters = 0L;
}

void CTomoStack::mSwap(int i, int k)
{
	if(i == k) return;
	//-----------------
	float* pfCent = m_ppfCenters[i];
	float* pfFrm = m_ppfFrames[i];
	float fTilt = m_pfTilts[i];
	int iAcq = m_piAcqIndices[i];
	int iSec = m_piSecIndices[i];
	//-----------------
	m_ppfCenters[i] = m_ppfCenters[k];
	m_ppfFrames[i] = m_ppfFrames[k];
	m_pfTilts[i] = m_pfTilts[k];
	m_piAcqIndices[i] = m_piAcqIndices[k];
	m_piSecIndices[i] = m_piSecIndices[k];
	//-----------------
	m_ppfCenters[k] = pfCent;
	m_ppfFrames[k] = pfFrm;
	m_pfTilts[k] = fTilt;
	m_piAcqIndices[k] = iAcq;
	m_piSecIndices[k] = iSec;
}

