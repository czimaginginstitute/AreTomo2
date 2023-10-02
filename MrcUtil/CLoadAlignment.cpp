#include "CMrcUtilInc.h"
#include "../CInput.h"
#include <Mrcfile/CMrcFileInc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>

using namespace MrcUtil;

CLoadAlignment* CLoadAlignment::m_pInstance = 0L;

CLoadAlignment* CLoadAlignment::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CLoadAlignment;
	return m_pInstance;
}

void CLoadAlignment::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CLoadAlignment::CLoadAlignment(void)
{
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
}

CLoadAlignment::~CLoadAlignment(void)
{
	mClean();
}

CAlignParam* CLoadAlignment::GetAlignParam(bool bClean)
{
	CAlignParam* pAlignParam = m_pAlignParam;
	if(bClean) m_pAlignParam = 0L;
	return pAlignParam;
}

CLocalAlignParam* CLoadAlignment::GetLocalParam(bool bClean)
{
	CLocalAlignParam* pLocalParam = m_pLocalParam;
	if(bClean) m_pLocalParam = 0L;
	return pLocalParam;
}

bool CLoadAlignment::DoIt(void)
{
	m_bFromAlnFile = false;
	m_bFromTiltRange = false;
	m_bFromAngFile = false;
	m_bFromHeader = false;
	mClean();
	//--------------------
	mReadHeaderSections();
	mDoAlnFile();
	bool bSuccess = mDoAngFile(); // Read angle file regardless
	if(!bSuccess) return false;
	//------------------------
	if(m_bFromAlnFile) return true;
	else if(m_bFromAngFile) return true;
	//----------------------------------
	mDoTiltRange();
	if(m_bFromTiltRange) return true;
	//-------------------------------
	mDoMrcFile();
	if(m_bFromHeader) return true;
	//----------------------------
	fprintf(stderr, "Error: tilt angles cannot be found.\n");
	return false;
}

void CLoadAlignment::mClean(void)
{
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
}

void CLoadAlignment::mDoAlnFile(void)
{
	CInput* pInput = CInput::GetInstance();
	//-----------------------------------------
	CLoadAlignFile loadAlignFile;
	loadAlignFile.DoIt(pInput->m_acAlnFile);
	m_pAlignParam = loadAlignFile.GetAlignParam(true);
	m_pLocalParam = loadAlignFile.GetLocalParam(true);
	//------------------------------------------------
	m_bFromAlnFile = (m_pAlignParam != 0L) ? true : false;
	if(m_bFromAlnFile) m_pAlignParam->SortByTilt();
}

void CLoadAlignment::mDoMrcFile(void)
{
	CInput* pInput = CInput::GetInstance();
	Mrc::CLoadMrc aLoadMrc;
	aLoadMrc.OpenFile(pInput->m_acInMrcFile);
	//---------------------------------------
	m_iNumSections = aLoadMrc.m_pLoadMain->GetSizeZ();
	int iNumFloats = aLoadMrc.m_pLoadMain->GetNumFloats();
	if(m_iNumSections <= 0 || iNumFloats <= 0) return;
	//------------------------------------------------
	if(m_pAlignParam == 0L) m_pAlignParam = new CAlignParam;
	m_pAlignParam->Create(m_iNumSections);
	//------------------------------------
	float fTilt = 0.0f, fMin = 100.0f, fMax = -100.0f;
	for(int i=0; i<m_iNumSections; i++)
	{	aLoadMrc.m_pLoadExt->DoIt(i);
		aLoadMrc.m_pLoadExt->GetTilt(&fTilt, 1);
		float fTiltAxis = aLoadMrc.m_pLoadExt->GetTiltAxis();
		if(pInput->m_afTiltAxis[0] != 0) 
		{	fTiltAxis = pInput->m_afTiltAxis[0];
		}
		m_pAlignParam->SetTilt(i, fTilt);
		m_pAlignParam->SetTiltAxis(i, fTiltAxis);
		m_pAlignParam->SetSecIndex(i, i);
		//-----------------------------
		if(fMin > fTilt) fMin = fTilt;
		else if(fMax < fTilt) fMax = fTilt;
	}
	m_pAlignParam->SetTiltRange(fMin, fMax);
	m_pAlignParam->SortByTilt();
	//--------------------------
	m_bFromHeader = true;
}

void CLoadAlignment::mDoTiltRange(void)
{	
	CInput* pInput = CInput::GetInstance();
	float fDiff = pInput->m_afTiltRange[1] - pInput->m_afTiltRange[0];
	if(fabs(fDiff) < 1) return;
	//-------------------------
	m_pAlignParam = new CAlignParam;
	m_pAlignParam->Create(m_iNumSections);
	//--------------------------------------------------------
	// Cannot have two tilts at the same angle.
	//--------------------------------------------------------
	float fStep = fDiff / (m_iNumSections - 1);
	for(int i=0; i<m_iNumSections; i++)
	{	float fTilt = pInput->m_afTiltRange[0] + i * fStep;
		m_pAlignParam->SetTilt(i, fTilt);
		m_pAlignParam->SetTiltAxis(i, pInput->m_afTiltAxis[0]);
		m_pAlignParam->SetSecIndex(i, i);
	}
	//-------------------------------------
	m_pAlignParam->SetTiltRange(pInput->m_afTiltRange[0], 
	   pInput->m_afTiltRange[1]);
	m_pAlignParam->SortByTilt();
	//------------------------
	m_bFromTiltRange = true;
}

bool CLoadAlignment::mDoAngFile(void)
{
	//-----------------------------------------------------------------
	// AngFile is a two-column text file where the 1st column is the
	// tilt angle listed in the same order as the tilt images in the
	// input MRC file. The 2nd column is optional and oontains the
	// 1-based index indiccating the acquisition order of the tilt
	// images. The 2nd column is required for dose weighting and
	// generation of csv file for Relion 4.
	//----------------------------------------------------------------- 
	CInput* pInput = CInput::GetInstance();
	CAcqSequence* pAcqSequence = CAcqSequence::GetInstance();
	pAcqSequence->ReadAngFile(pInput->m_acAngFile);
	if(pAcqSequence->m_iNumSections <= 0) return true;
	//------------------------------------------------------------------
	// NOTE: number of sections in MRC header must match that in the
	// angle file passed at the command line. If not quit.
	//------------------------------------------------------------------
	if(m_iNumSections != pAcqSequence->m_iNumSections)
	{	char acErr[256] = {'\0'};
		sprintf(acErr, "MRC file: %d images, angle file: %d images");
		fprintf(stderr, "Error: mismatch, %s.\n\n", acErr);
		return false;
	}
	//---------------------------------------------------------
	if(m_bFromAlnFile) 
	{	pAcqSequence->SortByAcquisition();
		m_bFromAngFile = false;
		return true;
	}	
	//-----------------------------
	m_pAlignParam = new CAlignParam;
	m_pAlignParam->Create(m_iNumSections);
	//-----------------------------------
	for(int i=0; i<pAcqSequence->m_iNumSections; i++)
	{	float fTilt = pAcqSequence->GetTiltAngle(i);
		m_pAlignParam->SetTilt(i, fTilt);
		m_pAlignParam->SetTiltAxis(i, pInput->m_afTiltAxis[0]); 
		m_pAlignParam->SetSecIndex(i, i);
	}
	m_pAlignParam->SortByTilt();
	//--------------------------------------------
	pAcqSequence->SortByAcquisition();
	m_bFromAngFile = true;
	return true;
}

void CLoadAlignment::mReadHeaderSections(void)
{
	CInput* pInput = CInput::GetInstance();
	Mrc::CLoadMrc aLoadMrc;
	aLoadMrc.OpenFile(pInput->m_acInMrcFile);
	m_iNumSections = aLoadMrc.m_pLoadMain->GetSizeZ();
}

