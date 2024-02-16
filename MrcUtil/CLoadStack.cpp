#include "CMrcUtilInc.h"
#include "../CInput.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

using namespace MrcUtil;

CLoadStack::CLoadStack(void)
{
	m_pTomoStack = 0L;
}

CLoadStack::~CLoadStack(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	m_pTomoStack = 0L;
}

CTomoStack* CLoadStack::DoIt(char* pcMrcFile)
{	
	CInput* pInput = CInput::GetInstance();
	m_aLoadMrc.OpenFile(pcMrcFile);
	//-----------------
	int iMode = m_aLoadMrc.m_pLoadMain->GetMode();
	m_aiStkSize[0] = m_aLoadMrc.m_pLoadMain->GetSizeX();
	m_aiStkSize[1] = m_aLoadMrc.m_pLoadMain->GetSizeY();
	m_aiStkSize[2] = m_aLoadMrc.m_pLoadMain->GetSizeZ();
	mPrintStackInfo(m_aiStkSize, iMode);
	//-----------------
	float fPixSize = m_aLoadMrc.GetPixelSize();
	if(fPixSize > 0 && pInput->m_fPixelSize <= 0)
	{	pInput->m_fPixelSize = fPixSize;
	}
	//-----------------	
	m_pTomoStack = new CTomoStack;
	m_pTomoStack->Create(m_aiStkSize);
	//-----------------
	mLoadTiltAngles();
	//-----------------
	printf("Start to load images.\n");
	if(iMode == Mrc::eMrcShort) mLoadShort();
	else if(iMode == Mrc::eMrcUShort) mLoadUShort();
	else if(iMode == Mrc::eMrcFloat) mLoadFloat();
	//-----------------
	m_aLoadMrc.CloseFile();
	CTomoStack* pTomoStack = m_pTomoStack;
	m_pTomoStack = 0L;
	return pTomoStack;
}

void CLoadStack::mLoadTiltAngles(void)
{
	int iNumFloats = m_aLoadMrc.m_pLoadMain->GetNumFloats();
	if(iNumFloats < 1) return;
	//-----------------
	float fTilt = 0.0f;
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	m_aLoadMrc.m_pLoadExt->DoIt(i);
		m_aLoadMrc.m_pLoadExt->GetTilt(&fTilt, 1);
		if(fabs(fTilt) > 75.0f) continue;
		//----------------
		m_pTomoStack->m_pfTilts[i] = fTilt;
	}
}

void CLoadStack::mLoadShort(void)
{
	int iPixels = m_aiStkSize[0] * m_aiStkSize[1];
	short* psBuf = new short[iPixels];
	//-----------------
	for(int i=0; i<m_aiStkSize[2]; i++)
        {       int iLeft = m_aiStkSize[2] - 1 - i;
		printf("...... Load image %3d, %3d left\n", i+1, iLeft);
		m_aLoadMrc.m_pLoadImg->DoIt(i, (char*)psBuf);
		//----------------
		float* pfFrame = m_pTomoStack->GetFrame(i);
		for(int j=0; j<iPixels; j++)
		{	pfFrame[j] = psBuf[j];
		}
		m_pTomoStack->m_piSecIndices[i] = i;
	}
	printf("All images have been loaded.\n\n");
	//-----------------
	if(psBuf != 0L) delete[] psBuf;
}

void CLoadStack::mLoadUShort(void)
{
	int iPixels = m_aiStkSize[0] * m_aiStkSize[1];
        unsigned short* pusBuf = new unsigned short[iPixels];
	//----------------- 
        for(int i=0; i<m_aiStkSize[2]; i++)
        {       int iLeft = m_aiStkSize[2] - 1 - i;
                printf("...... Load image %3d, %3d left\n", i+1, iLeft);
		m_aLoadMrc.m_pLoadImg->DoIt(i, (char*)pusBuf);
		//----------------
		float* pfFrame = m_pTomoStack->GetFrame(i);
                for(int j=0; j<iPixels; j++)
                {	pfFrame[j] = pusBuf[j];
                }
		m_pTomoStack->m_piSecIndices[i] = i;
        }
	printf("All images have been loaded.\n\n");
	//----------------- 
        if(pusBuf != 0L) delete[] pusBuf;
}

void CLoadStack::mLoadFloat(void)
{
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	int iLeft = m_aiStkSize[2] - 1 - i;
                printf("...... Load image %3d, %3d left\n", i+1, iLeft);
		float* pfFrame = m_pTomoStack->GetFrame(i);
		m_aLoadMrc.m_pLoadImg->DoIt(i, (char*)pfFrame);
		m_pTomoStack->m_piSecIndices[i] = i;
	}
	printf("All images have been loaded.\n\n");
}

void CLoadStack::mPrintStackInfo(int* piStkSize, int iMode)
{
	printf("MRC file size: %d  %d  %d\n", piStkSize[0],
	   piStkSize[1], piStkSize[2]);
	printf("MRC mode: %d\n", iMode);
}

