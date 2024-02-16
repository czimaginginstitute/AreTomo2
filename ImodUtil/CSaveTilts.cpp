#include "CImodUtilInc.h"
#include "../CInput.h"
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace ImodUtil;

CSaveTilts::CSaveTilts(void)
{
	m_iLineSize = 256;
	m_pcOrderedList = 0L;
	m_pvFile = 0L;
}

CSaveTilts::~CSaveTilts(void)
{
	if(m_pvFile != 0L) fclose((FILE*)m_pvFile);
	mClean();
}

void CSaveTilts::DoIt
(	MrcUtil::CAlignParam* pGlobalParam,
	const char* pcFileName
)
{	mClean();
	//-----------------
	FILE* pFile = fopen(pcFileName, "wt");
	if(pFile == 0L) return;
	//-----------------
	m_pvFile = pFile;
	m_pGlobalParam = pGlobalParam;
	//-----------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iOutImod == 1) mSaveForRelion();
	else if(pInput->m_iOutImod == 2) mSaveForWarp();
	else if(pInput->m_iOutImod == 3) mSaveForAligned();
	//-----------------
	fclose(pFile);
	m_pvFile = 0L;
}

//-----------------------------------------------------------------------------
// 1. -OutImod = 3 used for aligned tilt series where the tilt images are
//    ordered according to the tilt angles.
// 2. This is why we use MAM::CAlignParam to generate tilt angle list.
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForAligned(void)
{
	FILE* pFile = (FILE*)m_pvFile;
	int iLast = m_pGlobalParam->m_iNumFrames - 1;
        for(int i=0; i<=iLast; i++)
        {       float fTilt = m_pGlobalParam->GetTilt(i);
                fprintf(pFile, "%8.2f\n", fTilt);
        }
}

//-----------------------------------------------------------------------------
// 1. -OutImod = 2 used for dark-removed tilt series. Since this tilt series
//    gets saved after being sorted by tilt angle. We should use CAlignParam
//    as in mSaveForAligned.
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForWarp(void)
{
	this->mSaveForAligned();	
}

//-----------------------------------------------------------------------------
// Relion 4 requires the last line have a line return per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForRelion(void)
{
	mGenList();
	FILE* pFile = (FILE*)m_pvFile;
	//-----------------
	for(int i=0; i<m_iAllTilts; i++)
	{	char* pcLine = m_pcOrderedList + i * m_iLineSize;
		fprintf(pFile, "%s\n", pcLine);
	}
	//-----------------
	mClean();
}

//-----------------------------------------------------------------------------
// 1. Generate a ordered list of tilt angles sorted by the section indices
//    of tilt images in the input MRC file.
//-----------------------------------------------------------------------------
void CSaveTilts::mGenList(void)
{
        MrcUtil::CDarkFrames* pDarkFrames =
           MrcUtil::CDarkFrames::GetInstance();
        //-----------------
	m_iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	m_pcOrderedList = new char[m_iAllTilts * m_iLineSize];
	//-----------------
	for(int i=0; i<m_iAllTilts; i++)
	{	float fTilt = pDarkFrames->GetTilt(i);
		int iSecIdx = pDarkFrames->GetSecIdx(i);
		char* pcLine = m_pcOrderedList + iSecIdx * m_iLineSize;
		sprintf(pcLine, "%8.2f", fTilt);
	}
}

void CSaveTilts::mClean(void)
{
	if(m_pcOrderedList != 0L)
	{	delete[] m_pcOrderedList;
		m_pcOrderedList = 0L;
	}
}

