#pragma once

#include "ReconBase.h"
#include <Util/Util_Volume.h>

//====================================================================
// Dependent: CReconBase::SetProjs
//            CReconBase::SetVolume
//            CReconBase::SetProjIndices
//			  CReconBase::SetPrweightBox
// SetVolume: sets the volume data that will be projected.
// SetProjs:  provides the projection angles and the buffer that 
//            holds the projected data.
// SetProjIndices:
//            specifices a series indices to the projection angles 
//            at which the volume will be projected.
//====================================================================
class CForProj : public CReconBase
{
public:	
	CForProj(void);
	~CForProj(void);
	void SetPreweightBox(float* pfWBoxOrig, float* pfWBoxDim);
	void DoIt(int iStartY, int iSizeY);
private:
	void mConfigForProj(void);
	void* m_pvForProjImpl;
	bool m_bWeight;
};
