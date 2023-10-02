#include "CProjAlignInc.h"
#include <memory.h>
#include <stdio.h>

using namespace ProjAlign;

CParam* CParam::m_pInstance = 0L;

CParam* CParam::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CParam;
	return m_pInstance;
}

void CParam::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CParam::CParam(void)
{
	m_iIterations = 5;
	m_fTol = 0.5f;
	m_iVolZ = 256;
	m_afMaskSize[0] = 1.0f;
	m_afMaskSize[1] = 1.0f;
	m_fXcfSize = 1024.0f;
}

CParam::~CParam(void)
{
}

