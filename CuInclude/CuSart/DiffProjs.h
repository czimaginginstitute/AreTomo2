#pragma once

#include "ReconBase.h"
#include <Util/Util_Volume.h>

//====================================================================
// Dependent: CReconBase::SetProjs
//            CReconBase::SetVolume
//            CReconBase::SetRangeX 
//            CReconBase::SetPreweightBox
//            CReconBase::SetProjIndices
// Usage: 
// 1.  must call SetProjs, SetVolumeZ, SetRangeX, and SetPreweightBox 
//    first before calling DoIt to start the reconstruction.
// 2. SetRangeX, and SetPreweightBox are delcared in the
//    base class CReconBase
//
//====================================================================
class CDiffProjs : public CReconBase
{
public:
	CDiffProjs(void);
	~CDiffProjs(void);
	void SetVolumeZ(int iVolZ);
	void SetProjDiff(Util_Volume* pProjDiff);
	void DoIt(int iStartY, int iSizeY);
private:
	void mConfigBackProj(void);
	void mConfigDiffProj(void);
	void mCreateObjects(void);
	void mDeleteObjects(void);
	void mToHost(int iY);
	void* m_pvBackProjImpl;
	void* m_pvDiffProjImpl;
	int m_iVolZ;
	Util_Volume* m_pProjDiff;
	Util_Volume* m_pUtilVol;
};
