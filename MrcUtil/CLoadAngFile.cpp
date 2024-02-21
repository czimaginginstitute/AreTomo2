#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <queue>

using namespace MrcUtil;

CLoadAngFile::CLoadAngFile(void)
{
}

CLoadAngFile::~CLoadAngFile(void)
{
}

bool CLoadAngFile::DoIt
(	char* pcAngFile,
	CTomoStack* pTomoStack
)
{	m_bLoaded = false;
	//-----------------
	if(pcAngFile == 0L || strlen(pcAngFile) == 0) return false;
	FILE* pFile = fopen(pcAngFile, "r");
        if(pFile == 0L) return false;
	//-----------------
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
	//-----------------
	int iNumLines = lines.size();
	float* pfTilts = new float[iNumLines];
	int* piAcqIdxs = new int[iNumLines];
	int iCount = 0;
	//-----------------
	float fTilt = 0.0f;
	int iAcqIndex = -1;
	for(int i=0; i<iNumLines; i++)
	{	char* pcLine = lines.front();
		lines.pop();
		int iItems = sscanf(pcLine, "%f %d", &fTilt, &iAcqIndex);
		if(iItems > 0)
		{	pfTilts[iCount] = fTilt;
			if(iItems < 2) iAcqIndex = 0;
			piAcqIdxs[iCount] = iAcqIndex;
			iCount += 1;
		}
		delete[] pcLine;
	}
	//-----------------------------------------------
	// Make sure piAcqIdxs are 1-based.
	//-----------------------------------------------
	int iMinAcq = piAcqIdxs[0];
	for(int i=1; i<iCount; i++)
	{	if(piAcqIdxs[i] >= iMinAcq) continue;
		iMinAcq = piAcqIdxs[i];
	}
	for(int i=0; i<iCount; i++)
	{	piAcqIdxs[i] = piAcqIdxs[i] - iMinAcq + 1;
	}
	//-----------------
	if(iCount == pTomoStack->m_aiStkSize[2])
	{	pTomoStack->SetTilts(pfTilts);
		pTomoStack->SetAcqs(piAcqIdxs);
		m_bLoaded = true;
	}
	//-----------------
	if(pfTilts != 0L) delete[] pfTilts;
	if(piAcqIdxs != 0L) delete[] piAcqIdxs;
	//-----------------
	return m_bLoaded;
}

