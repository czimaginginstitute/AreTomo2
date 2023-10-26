#pragma once

#include "ReconBase.h"
#include "TomoSmooth.h"

//====================================================================
// Dependent: CReconBase::SetProjs
//            CReconBase::SetVolume
//            CReconBase::SetRangeX 
//            CReconBase::SetPreweightBox
//            CReconBase::SetProjIndices
//            CReconBase::SetNumSets
// SetVolume: provides the initial volume, which also stores later 
// the reconstructed volume.
//====================================================================
class CTomoSart : public CReconBase
{
public:
	CTomoSart(void);
	~CTomoSart(void);
	void SetSmooth(CTomoSmooth* pSmooth, unsigned int uiInterval)
	{	m_pTomoSmooth = pSmooth;
		m_uiSmoothInt = uiInterval;
	}
	void DoIt(int iStartY, int iSizeY);
	float* GetRfactor(void);	/* [out] */
	float GetRawRfactor(int iIter);
	float GetDifRfactor(int iIter);
	float* m_pfRfactor;  /* do not free */
protected:
	void mSmooth(void);
	void mClearRfactor(void);
	void* m_pvSartImpl;
	float m_fRelax[2];
	unsigned int m_uiSmoothInt;
	CTomoSmooth* m_pTomoSmooth;
};
