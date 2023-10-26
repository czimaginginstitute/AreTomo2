#pragma once

#include "TomoVolume.h"
#include <Util/Util_Volume.h>

//====================================================================
// 1. Each public method must be called. Only SetPrewightBox can be 
//    passed NULL.
//
//====================================================================
class CReconBase
{
public:	
	CReconBase(void);
	~CReconBase(void);
	void SetProjs(Util_Volume* pProjs, float* pfAngles);
	void SetVolume(CTomoVolume* pTomoVol);
	void SetRangeX(unsigned int* puiRangeX);
	virtual void SetPreweightBox(float* pfOrig, float* pfDim);
	void SetIterations(unsigned int uiIterations);
	void SetNumSets(unsigned int uiNumSets, unsigned int* puiSetSize);
	void SetRelax(float* pfRelax);
	void SetProjIndices
	(  unsigned int* piIndex, 
	   unsigned int iNumIndices
	);
	virtual void DoIt(int iStartY, int iSizeY);
protected:
	void mCreateProjsTex(void);
	void mCreateAnglesTex(void);
	void mCreateVolumeTex(void);
	void mCreateRangeTex(void);
	void mCreateProjIndexTex(void);
	int* mGetSetRange(int iNthSet);	// do not free
	void mSetProjTex(int iSliceY);
	void mSetVolTex(int iSliceY);
	void mGenWeights(bool bSart);
	void mCalcSetSize(void);
	void mCreateProjSlice(void);
	void mCreateVolSlice(void);
	void mDeleteProjSlice(void);
	void mDeleteVolSlice(void);
	void* m_pvTextures;
	void* m_pvSetSize;
	Util_Volume* m_pProjs;
	float* m_pfAngles;
	CTomoVolume* m_pTomoVol;
	unsigned int* m_puiRangeX;
	float* m_pfPrewOrig;
	float* m_pfPrewDim;
	unsigned int m_uiNumSets;
	unsigned int m_uiIterations;
	float* m_pfRelax;
	float* m_pfProjSlice;
	float* m_pfVolSlice;
	unsigned int* m_puiProjIndex;
	unsigned int m_uiNumProjIndices;
	unsigned int* m_puiSetSize;
	int m_iStartY;
	int m_iSizeY;
};
