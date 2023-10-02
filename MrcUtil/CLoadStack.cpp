#include "CMrcUtilInc.h"
#include "../CInput.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

using namespace MrcUtil;

CLoadStack* CLoadStack::m_pInstance = 0L;

CLoadStack* CLoadStack::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CLoadStack;
	return m_pInstance;
}

void CLoadStack::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CLoadStack::CLoadStack(void)
{
	m_pTomoStack = 0L;
}

CLoadStack::~CLoadStack(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	m_pTomoStack = 0L;
}

CTomoStack* CLoadStack::DoIt(CAlignParam* pAlignParam)
{	
	m_pAlignParam = pAlignParam;
	CInput* pInput = CInput::GetInstance();
	m_aLoadMrc.OpenFile(pInput->m_acInMrcFile);
	//-----------------------------------------
	int iMode = m_aLoadMrc.m_pLoadMain->GetMode();
	m_aiStkSize[0] = m_aLoadMrc.m_pLoadMain->GetSizeX();
	m_aiStkSize[1] = m_aLoadMrc.m_pLoadMain->GetSizeY();
	m_aiStkSize[2] = m_aLoadMrc.m_pLoadMain->GetSizeZ();
	mPrintStackInfo(m_aiStkSize, iMode);
	if(m_aiStkSize[2] > m_pAlignParam->m_iNumFrames)
	{	m_aiStkSize[2] = m_pAlignParam->m_iNumFrames;
	}
	//---------------------------------------------------
	float fPixSize = m_aLoadMrc.GetPixelSize();
	if(fPixSize > 0 && pInput->m_fPixelSize <= 0)
	{	pInput->m_fPixelSize = fPixSize;
	}
	//--------------------------------------	
	bool bAlloc = true;
	m_pTomoStack = new CTomoStack;
	m_pTomoStack->Create(m_aiStkSize, true);
	//--------------------------------------
	printf("Start to load images.\n");
	if(iMode == Mrc::eMrcShort) mLoadShort();
	else if(iMode == Mrc::eMrcUShort) mLoadUShort();
	else if(iMode == Mrc::eMrcFloat) mLoadFloat();
	//--------------------------------------------
	m_aLoadMrc.CloseFile();
	CTomoStack* pTomoStack = m_pTomoStack;
	m_pTomoStack = 0L;
	m_pAlignParam = 0L;
	return pTomoStack;
}

void CLoadStack::mLoadShort(void)
{
	int iPixels = m_aiStkSize[0] * m_aiStkSize[1];
	short* psBuf = new short[iPixels];
	//--------------------------------
	for(int i=0; i<m_pAlignParam->m_iNumFrames; i++)
        {       int iLeft = m_pAlignParam->m_iNumFrames - 1 - i;
		printf("...... Load image %3d, %3d left\n", i+1, iLeft);
		//------------------------------------------------------
		int iSecIdx = m_pAlignParam->GetSecIndex(i);
		m_aLoadMrc.m_pLoadImg->DoIt(iSecIdx, (char*)psBuf);
		//-------------------------------------------------
		float* pfFrame = m_pTomoStack->GetFrame(i);
		for(int j=0; j<iPixels; j++)
		{	pfFrame[j] = psBuf[j];
		}
	}
	printf("All images have been loaded.\n\n");
	//-----------------------------------------
	if(psBuf != 0L) delete[] psBuf;
}

void CLoadStack::mLoadUShort(void)
{
	int iPixels = m_aiStkSize[0] * m_aiStkSize[1];
        unsigned short* pusBuf = new unsigned short[iPixels];
        //---------------------------------------------------
        for(int i=0; i<m_pAlignParam->m_iNumFrames; i++)
        {       int iLeft = m_pAlignParam->m_iNumFrames - 1- i;
                printf("...... Load image %3d, %3d left\n", i+1, iLeft);
		//------------------------------------------------------
		int iSecIdx = m_pAlignParam->GetSecIndex(i);
		m_aLoadMrc.m_pLoadImg->DoIt(iSecIdx, (char*)pusBuf);
		//--------------------------------------------------
		float* pfFrame = m_pTomoStack->GetFrame(i);
                for(int j=0; j<iPixels; j++)
                {	pfFrame[j] = pusBuf[j];
                }
        }
	printf("All images have been loaded.\n\n");
        //-----------------------------------------
        if(pusBuf != 0L) delete[] pusBuf;
}

void CLoadStack::mLoadFloat(void)
{
	for(int i=0; i<m_pAlignParam->m_iNumFrames; i++)
	{	int iLeft = m_pAlignParam->m_iNumFrames - 1- i;
                printf("...... Load image %3d, %3d left\n", i+1, iLeft);
		//------------------------------------------------------
		int iSecIdx = m_pAlignParam->GetSecIndex(i);
		float* pfFrame = m_pTomoStack->GetFrame(i);
		m_aLoadMrc.m_pLoadImg->DoIt(iSecIdx, (char*)pfFrame);
	}
	printf("All images have been loaded.\n\n");
}

void CLoadStack::mPrintStackInfo(int* piStkSize, int iMode)
{
	printf("MRC file size: %d  %d  %d\n", piStkSize[0],
                piStkSize[1], piStkSize[2]);
        printf("MRC mode: %d\n", iMode);
	if(piStkSize[2] != m_pAlignParam->m_iNumFrames)
	{	printf("Alignment file has %d images\n", 
		   m_pAlignParam->m_iNumFrames);
		printf("Extra images are not loaded.\n");
	}
	printf("\n");
}

