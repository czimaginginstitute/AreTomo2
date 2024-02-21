#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace MrcUtil;

CSaveAlnFile::CSaveAlnFile(void)
{
	m_pvFile = 0L;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
}

CSaveAlnFile::~CSaveAlnFile(void)
{
	mCloseFile();
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
}

void CSaveAlnFile::DoIt
(	char* pcInMrcFile,
	char* pcOutMrcFile,
	CAlignParam* pAlignParam,
	CLocalAlignParam* pLocalParam
)
{	char acAlnFile[256] = {'\0'};
	strcpy(acAlnFile, pcOutMrcFile);
	char* pcOutSlash = strrchr(acAlnFile, '/');
	char* pcInSlash = strrchr(pcInMrcFile, '/');
	//------------------------------------------
	if(pcInSlash == 0L)
	{	if(pcOutSlash == 0L) strcpy(acAlnFile, pcInMrcFile);
		else strcat(pcOutSlash, pcInMrcFile);
	}
	else
	{	if(pcOutSlash == 0L) strcpy(acAlnFile, pcInSlash+1);
		else strcpy(pcOutSlash, pcInSlash);
	}
	char* pcMrc = strstr(acAlnFile, ".mrc");
	if(pcMrc == 0L) strcat(acAlnFile, ".aln");
	else strcpy(pcMrc, ".aln");
	//-------------------------
	FILE* pFile = fopen(acAlnFile, "wt");
	if(pFile == 0L)
	{	printf("Unable to open %s.\n", acAlnFile);
		printf("Alignment data will not be saved\n\n");
		return;
	}
	m_pvFile = pFile;
	//---------------
	m_pAlignParam = pAlignParam;
	m_pLocalParam = pLocalParam;
	mSaveHeader();
	mSaveGlobal();
	mSaveLocal();
	//-----------
	mCloseFile();
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
}

void CSaveAlnFile::mSaveHeader(void)
{
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	m_iNumTilts = 0; m_iNumPatches = 0;
	if(m_pAlignParam != 0L) m_iNumTilts = m_pAlignParam->m_iNumFrames;
	if(m_pLocalParam != 0L) m_iNumPatches = m_pLocalParam->m_iNumPatches;
	//------------------
	FILE* pFile = (FILE*)m_pvFile;
	fprintf(pFile, "# AreTomo Alignment / Priims bprmMn \n");
	fprintf(pFile, "# RawSize = %d %d %d\n", 
	   pDarkFrames->m_aiRawStkSize[0],
	   pDarkFrames->m_aiRawStkSize[1],
	   pDarkFrames->m_aiRawStkSize[2]);
	fprintf(pFile, "# NumPatches = %d\n", m_iNumPatches);
	//-----------------------------------------------
	// 1) Track section IDs of dark images here so
	// that we know which dark images are discarded
	// in the raw tilt series (iSecIdx).
	// 2) When tilt images are sorted by tilt angles,
	// iDarkFm shows which tilt image is dark.
	// 3) This info is needed when a tilt series 
	// needs to be reconstructed again without 
	// repeating the alignment process.
        //-----------------------------------------------
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	int iDarkFm = pDarkFrames->GetDarkIdx(i);
		int iSecIdx = pDarkFrames->GetSecIdx(iDarkFm);
		float fTilt = pDarkFrames->GetTilt(iDarkFm);
		fprintf(pFile, "# DarkFrame =  %4d %4d %8.2f\n", 
		   iDarkFm, iSecIdx, fTilt);
	}
}

void CSaveAlnFile::mSaveGlobal(void)
{
	if(m_pAlignParam == 0L) return;
	FILE* pFile = (FILE*)m_pvFile;
	fprintf( pFile, "# SEC     ROT         GMAG       "
	   "TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n");
	//--------------------------------------------------------------------
	float afShift[] = {0.0f, 0.0f};
	for(int i=0; i<m_iNumTilts; i++)
	{	int iSecIdx = m_pAlignParam->GetSecIdx(i);
		float fTilt = m_pAlignParam->GetTilt(i);
		float fTiltAxis = m_pAlignParam->GetTiltAxis(i);
		m_pAlignParam->GetShift(i, afShift);
		fprintf( pFile, "%5d  %9.4f  %9.5f  %9.3f  %9.3f  %7.2f  "
		   "%7.2f  %7.2f  %7.2f  %8.2f\n", iSecIdx, fTiltAxis, 
		   1.0f, afShift[0], afShift[1], 1.0f, 1.0f, 1.0f, 
		   0.0f, fTilt);
	}
}

void CSaveAlnFile::mSaveLocal(void)
{
	if(m_pLocalParam == 0L) return;
	FILE* pFile = (FILE*)m_pvFile;
	//----------------------------
	fprintf(pFile, "# Local Alignment\n");
	int iSize = m_iNumPatches * m_iNumTilts;
	for(int i=0; i<iSize; i++)
	{	int t = i / m_iNumPatches;
		int p = i % m_iNumPatches;
		fprintf(pFile, "%4d %3d %8.2f  %8.2f  %8.2f  %8.2f  %4.1f\n", 
		   t, p, m_pLocalParam->m_pfCoordXs[i], 
		   m_pLocalParam->m_pfCoordYs[i],
		   m_pLocalParam->m_pfShiftXs[i],
		   m_pLocalParam->m_pfShiftYs[i],
		   m_pLocalParam->m_pfGoodShifts[i]);
	}
}

void CSaveAlnFile::mCloseFile(void)
{
	if(m_pvFile == 0L) return;
	fclose((FILE*)m_pvFile);
	m_pvFile = 0L;
}
