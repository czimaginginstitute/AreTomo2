#include "CImodUtilInc.h"
#include "../CInput.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace ImodUtil;

CSaveXF::CSaveXF(void)
{
	m_pvFile = 0L;
	m_iLineSize = 256;
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
}

void CSaveXF::mSaveForAligned(void)
{
	FILE* pFile = (FILE*)m_pvFile;
        for(int i=0; i<m_pGlobalParam->m_iNumFrames; i++)
        {       fprintf(pFile, "%9.3f %9.3f %9.3f %9.3f ",
                   1.0f, 0.0f, 0.0f, 1.0f);
                fprintf(pFile, "%9.2f  %9.2f\n", 0.0f, 0.0f);
        }
        fprintf(pFile, "\n");
}

void CSaveXF::mSaveForWarp(void)
{
	FILE* pFile = (FILE*)m_pvFile;
	//-----------------
	float xshift_imod = 0.0f;
        float yshift_imod = 0.0f;
        float afShift[2] = {0.0f};
        float fD2R = 0.01745329f;
        //-----------------
        for(int i=0; i<m_pGlobalParam->m_iNumFrames; i++)
        {       float fTiltAxis = m_pGlobalParam->GetTiltAxis(i) * fD2R;
                float a11 = (float)cos(-fTiltAxis);
                float a12 = -(float)sin(-fTiltAxis);
                float a21 = (float)sin(-fTiltAxis);
                float a22 = (float)cos(-fTiltAxis);
                //----------------
                m_pGlobalParam->GetShift(i, afShift);
                xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
                yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
                //----------------
                fprintf(pFile, "%9.3f %9.3f %9.3f %9.3f ", a11, a12, a21, a22);
                fprintf(pFile, "%9.2f  %9.2f\n", xshift_imod, yshift_imod);
        }
	fprintf(pFile, "\n");
}

//--------------------------------------------------------------------
// 1. This is for -OutImod 1 that generates Imod files for Relion4.
// 2. Since Relion4 works on raw tilt series, lines in the .xf file
//    need to be in the same order as the raw tilt series.
// 3. Hence, lines in .xf file are sorted by iSecIdx to keep the same
//    order as the input MRC file.
//--------------------------------------------------------------------
void CSaveXF::mSaveForRelion(void)
{
	MrcUtil::CDarkFrames* pDarkFrames =
           MrcUtil::CDarkFrames::GetInstance();
        //-----------------
        int iAllTilts = pDarkFrames->m_aiRawStkSize[2];
        int iSize = iAllTilts * m_iLineSize;
        char* pcOrderedList = new char[iSize];
        memset(pcOrderedList, 0, sizeof(char) * iSize);
        //-----------------
        float xshift_imod = 0.0f;
        float yshift_imod = 0.0f;
        float afShift[2] = {0.0f};
        float fD2R = 0.01745329f;
        //-----------------
        for(int i=0; i<m_pGlobalParam->m_iNumFrames; i++)
        {       float fTiltAxis = m_pGlobalParam->GetTiltAxis(i) * fD2R;
                float a11 = (float)cos(-fTiltAxis);
                float a12 = -(float)sin(-fTiltAxis);
                float a21 = (float)sin(-fTiltAxis);
                float a22 = (float)cos(-fTiltAxis);
                //----------------
                m_pGlobalParam->GetShift(i, afShift);
                xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
                yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
                //----------------
                int iSecIdx = m_pGlobalParam->GetSecIdx(i);
                char* pcLine = pcOrderedList + iSecIdx * m_iLineSize;
                //----------------
                sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
                   a11, a12, a21, a22, xshift_imod, yshift_imod);
        }
        //---------------------------------------------------------
        // 1) For dark images their tilt axes are set to 0 degree
        // and their shifts are set to 0.
        //---------------------------------------------------------
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
        {       int iDarkFrm = pDarkFrames->GetDarkIdx(i);
                int iSecIdx = pDarkFrames->GetSecIdx(iDarkFrm);
                //----------------
                char* pcLine = pcOrderedList + iSecIdx * m_iLineSize;
                sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
                   1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }
        //-----------------
	FILE* pFile = (FILE*)m_pvFile;
        for(int i=0; i<iAllTilts; i++)
        {       char* pcLine = pcOrderedList + i * m_iLineSize;
                fprintf(pFile, "%s\n", pcLine);
        }
        //-----------------
        if(pcOrderedList != 0L) delete[] pcOrderedList;
}

