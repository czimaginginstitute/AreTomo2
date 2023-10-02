#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace MrcUtil;

CAlignFile::CAlignFile(void)
{
	m_pvFile = 0L;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
}

CAlignFile::~CAlignFile(void)
{
	mCloseFile();
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
}

CAlignParam* CAlignFile::GetAlignParam(bool bClean)
{
	CAlignParam* pAlignParam = m_pAlignParam;
	if(bClean) m_pAlignParam = 0L;
	return pAlignParam;
}

CLocalAlignParam* CAlignFile::GetLocalParam(bool bClean)
{
	CLocalAlignParam* pLocalParam = m_pLocalParam;
	if(bClean) m_pLocalParam = 0L;
	return pLocalParam;
}

void CAlignFile::Save
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

void CAlignFile::mSaveHeader(void)
{
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	m_iNumTilts = 0; m_iNumPatches = 0;
	if(m_pAlignParam != 0L) m_iNumTilts = m_pAlignParam->m_iNumFrames;
	if(m_pLocalParam != 0L) m_iNumPatches = m_pLocalParam->m_iNumPatches;
	//-------------------------------------------------------------------
	FILE* pFile = (FILE*)m_pvFile;
	fprintf(pFile, "# AreTomo Alignment / Priims bprmMn \n");
	fprintf(pFile, "# RawSize = %d %d %d\n", 
	   pDarkFrames->m_aiRawStkSize[0],
	   pDarkFrames->m_aiRawStkSize[1],
	   pDarkFrames->m_aiRawStkSize[2]);
	fprintf(pFile, "# DarkFrames = %d\n", pDarkFrames->m_iNumDarks);
	fprintf(pFile, "# NumPatches = %d\n", m_iNumPatches);
	//---------------------------------------------------
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	int iFrmIdx = pDarkFrames->GetFrmIdx(i);
		int iSecIdx = pDarkFrames->GetSecIdx(i);
		float fTilt = pDarkFrames->GetTilt(i);
		fprintf(pFile, "# Dark frame =  %4d %4d %8.2f\n", iFrmIdx,
		   iSecIdx, fTilt);
	}
}

void CAlignFile::mLoadHeader(void)
{
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
	//-----------------
	char acLine[256] = {'\0'}, acBuf[64] = {'\0'};
	FILE* pFile = (FILE*)m_pvFile;
	fgets(acLine, 256, pFile);
	//------------------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	fgets(acLine, 256, pFile);
	int aiRawSize[3] = {0};
	if(strstr(acLine, "RawSize") != 0L)
	{	sscanf(acLine, "%s %d %d %d", acBuf, &aiRawSize[0], 
		   &aiRawSize[1], &aiRawSize[2]);
		pDarkFrames->Setup(aiRawSize);
	}
	//------------------------------------
	int iNumDarks = 0;
	fgets(acLine, 256, pFile);
	if(strstr(acLine, "DarkFrames") != 0L)
	{	sscanf(acLine, "%s %d", acBuf, &iNumDarks);
	}
	//--------------------------------------------------
	m_iNumPatches = 0;
	fgets(acLine, 256, pFile);
	if(strstr(acLine, "NumPatches") != 0L)
	{	sscanf(acLine, "%s %d", acBuf, &m_iNumPatches);
	}
	//-----------------------------------------------------
	int iFrmIdx = 0, iSecIdx = 0, fTilt = 0.0f, iCount = 0;
	while(iCount < iNumDarks)
	{	fgets(acLine, 256, pFile);
		if(strstr(acLine, "Dark") == 0L) continue;
		else iCount += 1;
		sscanf(acLine, "%s %d %d %f", acBuf, 
		   &iFrmIdx, iSecIdx, &fTilt);
		pDarkFrames->Add(iFrmIdx, iSecIdx, fTilt);
	}
	//------------------------------------------------
	m_iNumTilts = aiRawSize[2] - iNumDarks;
	if(m_iNumTilts <= 0) return;
	m_pAlignParam = new CAlignParam;
	m_pAlignParam->Create(m_iNumTilts);
	//---------------------------------
	if(m_iNumPatches == 0) return;
	m_pLocalParam = new CLocalAlignParam;
	m_pLocalParam->Setup(m_iNumTilts, m_iNumPatches);
}  
	

void CAlignFile::mSaveGlobal(void)
{
	if(m_pAlignParam == 0L) return;
	FILE* pFile = (FILE*)m_pvFile;
	fprintf( pFile, "# SEC     ROT         GMAG       "
	   "TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n");
	//--------------------------------------------------------------------
	float afShift[] = {0.0f, 0.0f};
	for(int i=0; i<m_iNumTilts; i++)
	{	int iSecIdx = m_pAlignParam->GetSecIndex(i);
		float fTilt = m_pAlignParam->GetTilt(i);
		float fTiltAxis = m_pAlignParam->GetTiltAxis(i);
		m_pAlignParam->GetShift(i, afShift);
		fprintf( pFile, "%5d  %9.4f  %9.5f  %9.3f  %9.3f  %7.2f  "
		   "%7.2f  %7.2f  %7.2f  %8.2f\n", iSecIdx, fTiltAxis, 
		   1.0f, afShift[0], afShift[1], 1.0f, 1.0f, 1.0f, 
		   0.0f, fTilt);
	}
}

void CAlignFile::mSaveLocal(void)
{
	if(m_pLocalParam == 0L) return;
	FILE* pFile = (FILE*)m_pvFile;
	//----------------------------
	fprintf(pFile, "# Local Alignment\n");
	int iSize = m_iNumPatches * m_iNumTilts;
	for(int i=0; i<iSize; i++)
	{	int t = i / m_iNumPatches;
		int p = i % m_iNumPatches;
		fprintf(pFile, "%4d %3d %8.2f  %8.2f  %8.2f  %8.2f\n", t, p,
		   m_pLocalParam->m_pfCoordXs[i], 
		   m_pLocalParam->m_pfCoordYs[i],
		   m_pLocalParam->m_pfShiftXs[i],
		   m_pLocalParam->m_pfShiftYs[i]);
	}
}

bool CAlignFile::Load(char* pcFile)
{
	if(pcFile == 0L) return false;
	FILE* pFile = fopen(pcFile, "rt");
	if(pFile == 0L) return false;
	//---------------------------
	m_pvFile = pFile;
	mLoadHeader();
	if(m_iNumTilts == 0) return false;
	//--------------------------------
	mLoadGlobal();
	mLoadLocal();
	mCloseFile();
	return true;
}

void CAlignFile::mLoadGlobal(void)
{
	if(m_pAlignParam == 0L) return;
	//-----------------------------
	int iSecIndex = 0;
	float fTilt, fTiltAxis, afShift[2];
	float GMAG, SMEAN, SFIT, SCALE, BASE;
	//-----------------------------------
	FILE* pFile = (FILE*)m_pvFile;
	char acBuf[256];
	memset(acBuf, 0, sizeof(acBuf));
	fgets(acBuf, 256, pFile);
	//-----------------------
	for(int i=0; i<m_iNumTilts; i++)
	{	fgets(acBuf, 256, pFile);
		sscanf(acBuf, "%d  %f  %f  %f  %f  %f  %f  %f  %f  %f",
		   &iSecIndex, &fTiltAxis, &GMAG, afShift+0, afShift+1,
		   &SMEAN, &SFIT, &SCALE, &BASE, &fTilt);
		//---------------------------------------
		m_pAlignParam->SetSecIndex(i, iSecIndex);
		m_pAlignParam->SetTilt(i, fTilt);
		m_pAlignParam->SetTiltAxis(i, fTiltAxis);
		m_pAlignParam->SetShift(i, afShift);
	}
}

void CAlignFile::mLoadLocal(void)
{
	if(m_pLocalParam == 0L) return;
	//-----------------------------
	FILE* pFile = (FILE*)m_pvFile;
	char acBuf[256];
	fgets(acBuf, 256, pFile);
	//-----------------------
	int iSize = m_iNumTilts * m_iNumPatches;
	int t, p;
	for(int i=0; i<iSize; i++)
	{	fgets(acBuf, 256, pFile);
		sscanf(acBuf, "%d %d %f %f %f %f", &t, &p, 
		   m_pLocalParam->m_pfCoordXs+i,
		   m_pLocalParam->m_pfCoordYs+i, 
		   m_pLocalParam->m_pfShiftXs+i,
		   m_pLocalParam->m_pfShiftYs+i);
	}
}	

void CAlignFile::mCloseFile(void)
{
	if(m_pvFile == 0L) return;
	fclose((FILE*)m_pvFile);
	m_pvFile = 0L;
}
