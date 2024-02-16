#include "CMrcUtilInc.h"
#include "../CInput.h"
#include <Mrcfile/CMrcFileInc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>

using namespace MrcUtil;

CLoadMain* CLoadMain::m_pInstance = 0L;

CLoadMain* CLoadMain::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CLoadMain;
	return m_pInstance;
}

void CLoadMain::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CLoadMain::CLoadMain(void)
{
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
	m_pTomoStack = 0L;
}

CLoadMain::~CLoadMain(void)
{
	mClean();
}

CTomoStack* CLoadMain::GetTomoStack(bool bClean)
{
	CTomoStack* pTomoStack = m_pTomoStack;
	if(bClean) m_pTomoStack = 0L;
	return pTomoStack;
}

CAlignParam* CLoadMain::GetAlignParam(bool bClean)
{
	CAlignParam* pAlignParam = m_pAlignParam;
	if(bClean) m_pAlignParam = 0L;
	return pAlignParam;
}

CLocalAlignParam* CLoadMain::GetLocalParam(bool bClean)
{
	CLocalAlignParam* pLocalParam = m_pLocalParam;
	if(bClean) m_pLocalParam = 0L;
	return pLocalParam;
}

bool CLoadMain::DoIt(void)
{
	mClean();
	mLoadTomoStack();
	if(m_pTomoStack == 0L) 
	{	fprintf(stderr, "%s\n\n",
		   "Error: load tiltseries MRC file failed.");
		return false;
	}
	//-----------------
	mLoadTiltAngles();
	mCreateAlignParam();
	//----------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iAlign == 0) mLoadAlnFile();
	//----------------
	if(m_pTomoStack->bHasTiltAngles()) return true;
	fprintf(stderr, "Error: tilt angles cannot be found.\n");
	return false;
}

void CLoadMain::mClean(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
	m_pTomoStack = 0L;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
}

void CLoadMain::mLoadTomoStack(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	m_pTomoStack = 0L;
	//-----------------
	CLoadStack loadStack;
	CInput* pInput = CInput::GetInstance();
	m_pTomoStack = loadStack.DoIt(pInput->m_acInMrcFile);
	//-----------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	pDarkFrames->Setup(m_pTomoStack);
}

void CLoadMain::mLoadTiltAngles(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iAlign == 0) return;
	//-----------------
	mLoadAngFile();
	if(m_pTomoStack->bHasTiltAngles()) return;
	//--------------------------------------------------
	// Let's load from tilt range. This assumes tilt
	// images in MRC file are already sorted in either
	// ascending (from negative to positive tilts) or
	// descending order.
	//--------------------------------------------------
	float fDif = pInput->m_afTiltRange[1] -
	   pInput->m_afTiltRange[0];
	if(fabs(fDif) < 10.0f) return;
	//-----------------
	float fAngStep = fDif / (m_pTomoStack->m_aiStkSize[2] - 1);
	for(int i=0; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	m_pTomoStack->m_pfTilts[i] = i * fAngStep + 
		   pInput->m_afTiltRange[0];
	}
}

void CLoadMain::mCreateAlignParam(void)
{
	CInput* pInput = CInput::GetInstance();
        if(pInput->m_iAlign == 0) return;
	if(!m_pTomoStack->bHasTiltAngles()) return;
	//-----------------
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	m_pAlignParam = new CAlignParam;
	m_pAlignParam->Create(m_pTomoStack->m_aiStkSize[2]);
	for(int i=0; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	float fTilt = m_pTomoStack->m_pfTilts[i];
		int iSecIdx = m_pTomoStack->m_piSecIndices[i];
		m_pAlignParam->SetTilt(i, fTilt);
		m_pAlignParam->SetSecIdx(i, iSecIdx);
	}
	//-----------------
	m_pTomoStack->SortByTilt();
	m_pAlignParam->SortByTilt();
}

void CLoadMain::mLoadAngFile(void)
{
	CLoadAngFile loadAngFile;
	CInput* pInput = CInput::GetInstance();
	loadAngFile.DoIt(pInput->m_acAngFile, m_pTomoStack);
}

void CLoadMain::mLoadAlnFile(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iAlign != 0) return;
	//-----------------
	CLoadAlnFile loadAlnFile;
	bool bLoaded = loadAlnFile.DoIt(pInput->m_acAlnFile);
	if(!bLoaded) return;
	//-----------------
	m_pAlignParam = loadAlnFile.GetAlignParam(true);
	m_pLocalParam = loadAlnFile.GetLocalParam(true);
	//-----------------------------------------------
	// Sanity check: The raw size in CDarkFrames
	// set when .aln is loaded must match that
	// in CTomoStack. If not, loading fails.
	//-----------------------------------------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	if(pDarkFrames->m_aiRawStkSize[0] != 
	   m_pTomoStack->m_aiStkSize[0]) return;
	if(pDarkFrames->m_aiRawStkSize[1] != 
	   m_pTomoStack->m_aiStkSize[1]) return;
	if(pDarkFrames->m_aiRawStkSize[2] != 
	   m_pTomoStack->m_aiStkSize[2]) return;
	//-----------------------------------------------
	// Since m_pTomoStack is not sorted by tilt angle
	// yet, the images are in the same order as they
	// are in the MRC file. We can use the sec idx
	// in CAlignParam to fill the tilt angle.
	//-----------------------------------------------
	for(int i=0; i<m_pAlignParam->m_iNumFrames; i++)
	{	int iSecIdx = m_pAlignParam->GetSecIdx(i);
		float fTilt = m_pAlignParam->GetTilt(i);
		m_pTomoStack->m_pfTilts[iSecIdx] = fTilt;
	}
	//-----------------------------------------------
	// 1. Dark frames section indices and tilt angles
	// are loaded from .aln file.
	// 2. Now remove these dark frames
	//-----------------------------------------------
	for(int i=pDarkFrames->m_iNumDarks-1; i>=0; i--)
	{	int iSecIdx = pDarkFrames->GetDarkIdx(i);
		m_pTomoStack->RemoveFrame(iSecIdx);
	}
}

