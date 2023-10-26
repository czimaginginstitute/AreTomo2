#pragma once

#include "ReconBase.h"
#include "TomoSmooth.h"

//====================================================================
// 1. Must call SetProjs, SetVolume, SetRangeX, SetNumSets 
//    SetPreweightBox before calling DoIt to start the reconstruction.
// 2. SetVolume method provides the initial volume, which also stores 
//    later the reconstructed volume.
// 3. SetProjs, SetVolume, SetRangeX, and SetPreweightBox are 
//    delcared in the base class CReconBase
//====================================================================
class CTomoEm : public CReconBase
{
public:
	CTomoEm(void);
	~CTomoEm(void);
	void SetSmooth(CTomoSmooth* pSmooth, unsigned int uiInterval)
	{	m_pTomoSmooth = pSmooth;
		m_uiSmoothInt = uiInterval;
	}
	void DoIt(int iStartY, int iSizeY);
	float* GetRfactor(void);		/* [out] */
	float GetRawRfactor(int iIter);
	float GetDifRfactor(int iIter);
	float* m_pfRfactor;  /* do not free */
private:
	void mSmooth(void);
	void mClearRfactor(void);
	void* m_pvEmImpl;
	unsigned int m_uiSmoothInt;
	CTomoSmooth* m_pTomoSmooth;
};
