#pragma once

#include "ReconBase.h"

//====================================================================
// Dependent: CReconBase::SetProjs
//            CReconBase::SetVolume
//            CReconBase::SetRangeX 
//            CReconBase::SetProjIndices
// 
// SetVolume method provides the initial volume, which also stores 
// later the reconstructed volume.
//====================================================================
class CBackProj : public CReconBase
{
public:
	CBackProj(void);
	~CBackProj(void);
	void SetPreweightBox(float* pfOrig, float* pfDim);
	void DoIt(int iStartY, int iSizeY);
private:
	void mCreateObjects(void);
	void mDeleteObjects(void);
	void mConfigBackProj(void);
	void mConfigDiffProj(void);
	void* m_pvBackProjImpl;
	void* m_pvDiffProjImpl;
	bool m_bWeight;
};
