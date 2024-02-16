#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <string.h>

using namespace MrcUtil;

CDarkFrames* CDarkFrames::m_pInstance = 0L;

CDarkFrames* CDarkFrames::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CDarkFrames;
	return m_pInstance;
}

void CDarkFrames::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CDarkFrames::CDarkFrames(void)
{
	memset(m_aiRawStkSize, 0, sizeof(m_aiRawStkSize));
	m_iNumDarks = 0;
	m_piAcqIdxs = 0L;
	m_piSecIdxs = 0L;
	m_pfTilts = 0L;
	m_piDarkIdxs = 0L;
	m_pbDarkImgs = 0L;
}

CDarkFrames::~CDarkFrames(void)
{
	mClean();
}

void CDarkFrames::mClean(void)
{
	if(m_piAcqIdxs != 0L) delete[] m_piAcqIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_pbDarkImgs != 0L) delete[] m_pbDarkImgs;
	m_piAcqIdxs = 0L; 
	m_piSecIdxs = 0L;
	m_piDarkIdxs = 0L;
	m_pfTilts = 0L;
	m_pbDarkImgs = 0L;
}

void CDarkFrames::Setup(CTomoStack* pTomoStack)
{
	mClean();
	m_iNumDarks = 0;
	//-----------------
	memcpy(m_aiRawStkSize, pTomoStack->m_aiStkSize, sizeof(int) * 3);
	m_piAcqIdxs = new int[3 * m_aiRawStkSize[2]];
	m_piSecIdxs = &m_piAcqIdxs[m_aiRawStkSize[2]];
	m_piDarkIdxs = &m_piAcqIdxs[2 * m_aiRawStkSize[2]];
	//-----------------
	m_pfTilts = new float[m_aiRawStkSize[2]];
	memset(m_pfTilts, 0, sizeof(float) * m_aiRawStkSize[2]);
	//-----------------
	m_pbDarkImgs = new bool[m_aiRawStkSize[2]];
	memset(m_pbDarkImgs, 0, sizeof(bool) * m_aiRawStkSize[2]);
	//-----------------
	size_t tBytes = sizeof(int) * m_aiRawStkSize[2];
	memcpy(m_piAcqIdxs, pTomoStack->m_piAcqIndices, tBytes);
	memcpy(m_piSecIdxs, pTomoStack->m_piSecIndices, tBytes);
	//-----------------
	tBytes = sizeof(float) * m_aiRawStkSize[2];
	memcpy(m_pfTilts, pTomoStack->m_pfTilts, tBytes);
	//-----------------
	memset(m_pbDarkImgs, 0, sizeof(bool) * m_aiRawStkSize[2]);
}

void CDarkFrames::AddDark(int iFrmIdx)
{
	m_pbDarkImgs[iFrmIdx] = true;
	m_piDarkIdxs[m_iNumDarks] = iFrmIdx;
	m_iNumDarks += 1;
}

void CDarkFrames::AddTiltOffset(float fTiltOffset)
{
	for(int i=0; i<m_iNumDarks; i++)
	{	m_pfTilts[i] += fTiltOffset;
	}
}

int CDarkFrames::GetAcqIdx(int iFrame)
{
	return m_piAcqIdxs[iFrame];
}

int CDarkFrames::GetSecIdx(int iFrame)
{
	return m_piSecIdxs[iFrame];
}

float CDarkFrames::GetTilt(int iFrame)
{
	return m_pfTilts[iFrame];
}

int CDarkFrames::GetDarkIdx(int iNthDark)
{
	return m_piDarkIdxs[iNthDark];
}

bool CDarkFrames::IsDarkFrame(int iFrame)
{
	return m_pbDarkImgs[iFrame];
}

void CDarkFrames::GenImodExcludeList(char* pcLine, int iSize)
{
	if(m_iNumDarks <= 0) return;
	//-----------------
	strcpy(pcLine, "EXCLUDELIST ");
	char acBuf[16] = {'\0'};
	int iLast = m_iNumDarks - 1;
	for(int i=0; i<iLast; i++)
	{	int iDarkFm = m_piDarkIdxs[i];
		int iSecIdx = m_piSecIdxs[iDarkFm] + 1;
		sprintf(acBuf, "%d,", iSecIdx);
		strcat(pcLine, acBuf); // Relion 1-based index
	}
	//-----------------
	int iDarkFm = m_piDarkIdxs[iLast];
	int iSecIdx = m_piSecIdxs[iDarkFm] + 1;
	sprintf(acBuf, "%d", iSecIdx);
	strcat(pcLine, acBuf);
}
