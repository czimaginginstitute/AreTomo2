#include "CStreAlignInc.h"
#include <memory.h>
#include <stdio.h>

using namespace Strealign;

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
	m_fBFactor = 100.0f;
	m_iIterations = 1;
	m_fTol = 0.5f;
	m_fTiltAxis = 0.0f;
}

CParam::~CParam(void)
{
}

