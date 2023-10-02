#include "CImodUtilInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace ImodUtil;

CSaveXF::CSaveXF(void)
{
}

CSaveXF::~CSaveXF(void)
{
}

void CSaveXF::DoIt
(	MrcUtil::CAlignParam* pGlobalParam, 
	const char* pcFileName
)
{	FILE* pFile = fopen(pcFileName, "wt");
	if(pFile == 0L) return;
	m_pvFile = pFile;
	m_pGlobalParam = pGlobalParam;
	//----------------------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iOutImod == 1) mSaveForRelion();
	else if(pInput->m_iOutImod == 2) mSaveForWarp();
	else if(pInput->m_iOutImod == 3) mSaveForAligned();
	fclose(pFile);
}

void CSaveXF::mSaveForAligned(void)
{
	// for dark removed aligned tilt series
	FILE* pFile = (FILE*)m_pvFile;
	int iLast = m_pGlobalParam->m_iNumFrames - 1;
	for(int i=0; i<m_pGlobalParam->m_iNumFrames; i++)
	{	fprintf(pFile, "%9.3f %9.3f %9.3f %9.3f ",
		   1.0f, 0.0f, 0.0f, 1.0f);
		fprintf(pFile, "%9.2f  %9.2f\n", 0.0f, 0.0f);
	}
	fprintf(pFile, "\n");
}

void CSaveXF::mSaveForWarp(void)
{
	FILE* pFile = (FILE*)m_pvFile;
	float fD2R = 3.141592654f / 180.0f;
	float afShift[2], fTiltAxis, fCos, fSin;
	float a11, a12, a21, a22, xshift_imod, yshift_imod;
	int iLast = m_pGlobalParam->m_iNumFrames - 1;
	for(int i=0; i<m_pGlobalParam->m_iNumFrames; i++)
	{	fTiltAxis = m_pGlobalParam->GetTiltAxis(i) * fD2R;
                a11 = (float)cos(-fTiltAxis);
                a12 = -(float)sin(-fTiltAxis);
                a21 = (float)sin(-fTiltAxis);
                a22 = (float)cos(-fTiltAxis);
		//---------------------------
		m_pGlobalParam->GetShift(i, afShift);
                xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
                yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
		//------------------------------------------------------
		fprintf(pFile, "%9.3f %9.3f %9.3f %9.3f ", a11, a12, a21, a22);
		fprintf(pFile, "%9.2f  %9.2f\n", xshift_imod, yshift_imod);
        }
	fprintf(pFile, "\n");
}

void CSaveXF::mSaveForRelion(void)
{
	FILE* pFile = (FILE*)m_pvFile;
	MrcUtil::CDarkFrames* pDarkFrames = 0L;
	pDarkFrames = MrcUtil::CDarkFrames::GetInstance();
	int iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	//---------------------------------------------
	int iLineSize = 256;
	char* pcLines = new char[iAllTilts * iLineSize];
	memset(pcLines, 0, sizeof(char) * iAllTilts * iLineSize);
	//-------------------------------------------------------
	float fD2R = 3.141592654f / 180.0f;
	float afShift[2], fTiltAxis, fCos, fSin;
	float a11, a12, a21, a22;
	float xshift_imod, yshift_imod;
	for(int i=0; i<m_pGlobalParam->m_iNumFrames; i++)
	{	fTiltAxis = m_pGlobalParam->GetTiltAxis(i) * fD2R;
		a11 = (float)cos(-fTiltAxis);
		a12 = -(float)sin(-fTiltAxis);
		a21 = (float)sin(-fTiltAxis);
		a22 = (float)cos(-fTiltAxis);
		//---------------------------
		m_pGlobalParam->GetShift(i, afShift);
		xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
		yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
		//------------------------------------------------------
		int iSecIdx = m_pGlobalParam->GetSecIndex(i);
		char* pcLine = pcLines + iSecIdx * iLineSize;
		//-------------------------------------------
		sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
		   a11, a12, a21, a22, xshift_imod, yshift_imod);
	}
	//-------------------------------------------------------
	a11 = 1.0f; a12 = 0.0f; a21 = 0.0f; a22 = 1.0f;
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	int iSecIdx = pDarkFrames->GetSecIdx(i);
		char* pcLine = pcLines + iSecIdx * iLineSize;
		sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
		   1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
	}
	//---------------------------------------------
	for(int i=0; i<iAllTilts; i++)		
	{	char* pcLine = pcLines + i * iLineSize;
		fprintf(pFile, "%s\n", pcLine);
	}
	fprintf(pFile, "\n");
	delete[] pcLines; 
}

