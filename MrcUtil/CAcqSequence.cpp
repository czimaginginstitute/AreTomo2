#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <queue>

using namespace MrcUtil;

CAcqSequence* CAcqSequence::m_pInstance = 0L;

CAcqSequence* CAcqSequence::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CAcqSequence;
	return m_pInstance;
}

void CAcqSequence::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CAcqSequence::CAcqSequence(void)
{
	m_pfTiltAngles = 0L;
	m_piAcqIndices = 0L;
	m_piSecIndices = 0L;
	m_iNumSections = 0;
}

CAcqSequence::~CAcqSequence(void)
{
	this->Clean();
}

void CAcqSequence::Clean(void)
{
	if(m_pfTiltAngles != 0L) delete[] m_pfTiltAngles;
	if(m_piAcqIndices != 0L) delete[] m_piAcqIndices;
	if(m_piSecIndices != 0L) delete[] m_piSecIndices;
	m_pfTiltAngles = 0L;
	m_piAcqIndices = 0L;
	m_piSecIndices = 0L;
	m_iNumSections = 0;
}

void CAcqSequence::ReadAngFile(char* pcAngFile)
{
	this->Clean();
	//------------
	if(pcAngFile == 0L || strlen(pcAngFile) == 0) return;
	FILE* pFile = fopen(pcAngFile, "r");
        if(pFile == 0L) return;
	//---------------------
	std::queue<char*> lines;
	while(!feof(pFile))
	{	char* pcLine = new char[256];
		memset(pcLine, 0, sizeof(char) * 256);
		fgets(pcLine, 256, pFile);
		if(strlen(pcLine) == 0 || pcLine[0] == '#')
		{	delete pcLine;
			continue;
		}
		else lines.push(pcLine);
	}
	fclose(pFile);
	//------------------------------
	int iNumLines = lines.size();
	m_pfTiltAngles = new float[iNumLines];
	m_piAcqIndices = new int[iNumLines];
	m_piSecIndices = new int[iNumLines];
	m_iNumSections = 0;
	//----------------------------------
	float fTilt = 0.0f;
	int iAcqIndex = -1;
	for(int i=0; i<iNumLines; i++)
	{	char* pcLine = lines.front();
		lines.pop();
		int iItems = sscanf(pcLine, "%f %d", &fTilt, &iAcqIndex);
		if(iItems > 0)
		{	m_pfTiltAngles[m_iNumSections] = fTilt;
			if(iItems < 2) iAcqIndex = -1;
			m_piAcqIndices[m_iNumSections] = iAcqIndex;
			m_piSecIndices[m_iNumSections] = m_iNumSections;
			m_iNumSections += 1;
		}
		delete[] pcLine;
	}
	if(m_piAcqIndices[0] == -1)
	{	delete m_piAcqIndices;
		m_piAcqIndices = 0L;
	}
}

void CAcqSequence::AddTiltOffset(float fTiltOffset)
{
	if(m_iNumSections == 0) return;
        for(int i=0; i<m_iNumSections; i++)
        {       m_pfTiltAngles[i] += fTiltOffset;
        }
}

int CAcqSequence::GetAcqIndexFromTilt(float fTilt)
{
	if(m_piAcqIndices == 0L) return -1;
	else if(m_iNumSections <= 0) return -1;
	//-------------------------------------
	float fMin = (float)fabs(fTilt - m_pfTiltAngles[0]);
	int iMin = 0;
	for(int i=1; i<m_iNumSections; i++)
	{	float fDif = (float)fabs(fTilt - m_pfTiltAngles[i]);
		if(fDif >= fMin) continue;
		fMin = fDif;
		iMin = i;
	}
	return m_piAcqIndices[iMin];
}

int CAcqSequence::GetAcqIndexFromSection(int iSecIdx)
{
	if(m_piAcqIndices == 0L) return -1;
	else if(m_iNumSections <= 0) return -1;
	//-------------------------------------
	for(int i=0; i<m_iNumSections; i++)
	{	if(m_piSecIndices[i] != iSecIdx) continue;
		else return m_piSecIndices[i];
	}
	return -1;
}

int CAcqSequence::GetAcqIndex(int iEntry)
{
	if(m_piAcqIndices == 0L) return -1;
	else return m_piAcqIndices[iEntry];
}

int CAcqSequence::GetSecIndex(int iEntry)
{
	return m_piSecIndices[iEntry];
}

float CAcqSequence::GetTiltAngle(int iEntry)
{
	return m_pfTiltAngles[iEntry];
}

void CAcqSequence::SortByAcquisition(void)
{
	if(m_piAcqIndices == 0L) return;
	//------------------------------
	for(int i=0; i<m_iNumSections; i++)
	{	int iMin = m_piAcqIndices[i];
		int iMinIdx = i;
		for(int j=i+1; j<m_iNumSections; j++)
		{	if(m_piAcqIndices[j] >= iMin) continue;
			iMin = m_piAcqIndices[j];
			iMinIdx = j;
		}
		int iAcqIdx = m_piAcqIndices[i];
		int iSecIdx = m_piSecIndices[i];
		float fTemp = m_pfTiltAngles[i];
		m_piAcqIndices[i] = m_piAcqIndices[iMinIdx];
		m_piSecIndices[i] = m_piSecIndices[iMinIdx];
		m_pfTiltAngles[i] = m_pfTiltAngles[iMinIdx];
		m_piAcqIndices[iMinIdx] = iAcqIdx;
		m_piSecIndices[iMinIdx] = iSecIdx;
		m_pfTiltAngles[iMinIdx] = fTemp;
	}
}

bool CAcqSequence::hasSequence(void)
{
	if(m_piAcqIndices == 0L) return false;
	else return true;
}

void CAcqSequence::SaveCsv(char* pcCsvName, int iOutImod)
{	
	if(m_piAcqIndices == 0L) return;
	if(pcCsvName == 0L || strlen(pcCsvName) == 0) return;
	if(iOutImod == 1) mSaveCsvRelion(pcCsvName);
	else mSaveCsvWarp(pcCsvName);
}

void CAcqSequence::mSaveCsvRelion(char* pcCsvName)
{
	FILE* pFile = fopen(pcCsvName, "w");
	for(int i=0; i<m_iNumSections; i++)
	{	fprintf(pFile, "%5d, %8.2f\n", m_piAcqIndices[i],
		   m_pfTiltAngles[i]);
	}
	fclose(pFile);	
}

void CAcqSequence::mSaveCsvWarp(char* pcCsvName)
{
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	FILE* pFile = fopen(pcCsvName, "w");
	for(int i=0; i<m_iNumSections; i++)
	{	bool bDark = pDarkFrames->IsDarkSection(m_piSecIndices[i]);
		if(bDark) continue;
		fprintf(pFile, "%4d, %.2f\n", m_piAcqIndices[i],
		   m_pfTiltAngles[i]);
	}
	fclose(pFile);	
}
