#include "CImodUtilInc.h"
#include "../CInput.h"
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace ImodUtil;

CSaveCsv::CSaveCsv(void)
{
}

CSaveCsv::~CSaveCsv(void)
{
}

void CSaveCsv::DoIt
(	MrcUtil::CAlignParam* pGlobalParam,
	const char* pcFileName
)
{	FILE* pFile = fopen(pcFileName, "wt");
	if(pFile == 0L) return;
	m_pvFile = pFile;
	m_pGlobalParam = pGlobalParam;
	//---------------------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iOutImod == 1) mSaveForRelion();
	else if(pInput->m_iOutImod == 2) mSaveForWarp();
	else if(pInput->m_iOutImod == 3) mSaveForAligned();
	fclose(pFile);
}

//-----------------------------------------------------------------------------
// Relion 4 requires line return at the last line per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveCsv::mSaveForAligned(void)
{
	FILE* pFile = (FILE*)m_pvFile;
	fprintf(pFile, "ImageNumber, TiltAngle\n");
	// aligned & dark-removed tilt series
	int iLast = m_pGlobalParam->m_iNumFrames - 1;
	for(int i=0; i<=iLast; i++)
	{	int iSecIdx = m_pGlobalParam->GetSecIndex(i);
		float fTilt = m_pGlobalParam->GetTilt(i);
		fprintf(pFile, "%4d, %8.2f\n", iSecIdx+1, fTilt);
	}
}

void CSaveCsv::mSaveForWarp(void)
{
	this->mSaveForAligned();	
}

//-----------------------------------------------------------------------------
// Relion 4 requires the last line have a line return per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveCsv::mSaveForRelion(void)
{
	FILE* pFile = (FILE*)m_pvFile;
	fprintf(pFile, "ImageNumber, TiltAngle\n");
	// raw tilt series as input to Relion 4
	MrcUtil::CDarkFrames* pDarkFrames = 0L;
	pDarkFrames = MrcUtil::CDarkFrames::GetInstance();
	int iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	char* pcLines = new char[iAllTilts * 256];
	for(int i=0; i<m_pGlobalParam->m_iNumFrames; i++)
	{	float fTilt = m_pGlobalParam->GetTilt(i);
		int iSecIdx = m_pGlobalParam->GetSecIndex(i);
		char* pcLine = pcLines + iSecIdx * 256;
		sprintf(pcLine, "%4d, %8.2f", iSecIdx+1, fTilt);
	}
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	float fTilt = pDarkFrames->GetTilt(i);
		int iSecIdx = pDarkFrames->GetSecIdx(i);
		char* pcLine = pcLines + iSecIdx * 256;
		sprintf(pcLine, "%4d, %8.2f", iSecIdx+1, fTilt);
	}
	//------------------------------------------------------
	int iLast = iAllTilts - 1;
	for(int i=0; i<=iLast; i++)
	{	char* pcLine = pcLines + i * 256;
		fprintf(pFile, "%s\n", pcLine);
	}
	delete[] pcLines;
}
