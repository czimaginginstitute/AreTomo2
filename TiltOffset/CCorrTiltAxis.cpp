#include "CAlignInc.h"
#include "../../Util/CUtilInc.h"
#include "../../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace TomoAlign;

CCorrTiltAxis::CCorrTiltAxis(void)
{
	m_fFillVal = (float)-1e30;
}

CCorrTiltAxis::~CCorrTiltAxis(void)
{
}

void CCorrTiltAxis::DoIt
(	::MrcUtil::CTomoStack* pTomoStack,
	CStackShift* pStackShift,
	float fTiltAxis
)
{	m_pTomoStack = pTomoStack;
	m_pStackShift = pStackShift;
	m_fTiltAxis = fTiltAxis;
	//----------------------
	for(int i=0; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	printf("...... Rotate frame %4d, %4d left\n",
			i+1, pTomoStack->m_aiStkSize[2]-1-i);
		mGetValidRect(i);
		mRotateFrame(i);
	}
	printf("\n");
}

void CCorrTiltAxis::mGetValidRect(int iFrame)
{
	int iImageX = m_pTomoStack->m_aiStkSize[0];
	int iImageY = m_pTomoStack->m_aiStkSize[1];
	m_aiValidRect[0] = 0;
	m_aiValidRect[1] = 0;
	m_aiValidRect[2] = iImageX - 1;
	m_aiValidRect[3] = iImageY - 1;
	//-----------------------------
	float afShift[2] = {0.0f};
	m_pStackShift->GetShift(iFrame, afShift);
	//---------------------------------------
	if(afShift[0] > 0)
	{	m_aiValidRect[2] = (int)(iImageX - afShift[0] - 0.5f);
	}
	else
	{	m_aiValidRect[0] = (int)(-afShift[0] + 0.5f);
	}
	if(afShift[1] > 0)
	{	m_aiValidRect[3] = (int)(iImageY - afShift[1] - 0.5f);
	}
	else
	{	m_aiValidRect[1] = (int)(-afShift[1] + 0.5f);
	}
}

void CCorrTiltAxis::mRotateFrame(int iFrame)
{
	bool bGpu = true;
	float* pfFrame = m_pTomoStack->GetFrame(iFrame, false);
	//-----------------------------------------------------
	::Util::GRotate2D aRot2D;
	aRot2D.SetValidRect
	(  m_aiValidRect[0], m_aiValidRect[1],
	   m_aiValidRect[2], m_aiValidRect[3]
	);
	aRot2D.SetFillValue(m_fFillVal);
	aRot2D.SetImage(pfFrame, m_pTomoStack->m_aiStkSize, !bGpu);
	aRot2D.DoIt(-m_fTiltAxis, pfFrame, !bGpu);
	aRot2D.Clear();
}

