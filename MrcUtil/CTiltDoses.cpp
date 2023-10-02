#include "CMrcUtilInc.h"
#include "../CInput.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace MrcUtil;

CTiltDoses* CTiltDoses::m_pInstance = 0L;

CTiltDoses* CTiltDoses::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CTiltDoses;
	return m_pInstance;
}

void CTiltDoses::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CTiltDoses::CTiltDoses(void)
{
	m_pfDoses = 0L;
	m_bDoseWeight = false;
}

CTiltDoses::~CTiltDoses(void)
{
	this->Clean();
}

void CTiltDoses::Clean(void)
{
	if(m_pfDoses != 0L) delete[] m_pfDoses;
	m_pfDoses = 0L;
}

//-------------------------------------------------------------------
// This class stores the accumulated dose on each tilt image after
// dark images are removed.
//-------------------------------------------------------------------
void CTiltDoses::Setup(CAlignParam* pAlignParam)
{
	this->Clean();
	m_bDoseWeight = false;
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_fImgDose <= 0) return;
	CAcqSequence* pAcqSequence = CAcqSequence::GetInstance();
	if(!pAcqSequence->hasSequence()) return;
	//--------------------------------------
	m_bDoseWeight = true;
	m_iNumImgs = pAlignParam->m_iNumFrames;
	m_pfDoses = new float[m_iNumImgs];
	for(int i=0; i<m_iNumImgs; i++)
	{	int iSecIdx = pAlignParam->GetSecIndex(i);
		int iAcqIdx = pAcqSequence->GetAcqIndexFromSection(iSecIdx);
		m_pfDoses[i] = iAcqIdx * pInput->m_fImgDose;
	}
}

float* CTiltDoses::GetDoses(void) // do not free
{
	return m_pfDoses;
}

float CTiltDoses::GetDose(int iImage)
{
	if(m_pfDoses == 0L) return 0.0f;
	else return m_pfDoses[iImage];
}
