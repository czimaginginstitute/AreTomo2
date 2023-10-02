#pragma once

#include "../MrcUtil/CMrcUtilInc.h"

namespace MassNorm
{
class GFlipInt2D
{
public:
	GFlipInt2D(void);
	~GFlipInt2D(void);
	void DoIt
	( float* gfImg, int* piSize, bool bPadded,
	  float fMin, float fMax, cudaStream_t stream = 0
	);
};

class CLinearNorm
{
public:
	CLinearNorm(void);
	~CLinearNorm(void);
	void DoIt
	( MrcUtil::CTomoStack* pTomoStack,
	  MrcUtil::CAlignParam* pAlignParam
	);
	void FlipInt
	( MrcUtil::CTomoStack* pTomoStack
	);
private:
	float mCalcMean(int iFrame);
	void mSmooth(float* pfMeans);
	void mScale(int iFrame, float fScale);
	void mFlipInt(int iFrame);
	MrcUtil::CTomoStack* m_pTomoStack;
	int m_aiStart[2];
	int m_aiSize[2];
	int m_iNumFrames;
	float m_fMissingVal;
	float m_fRefMean;
};

class CPositivity
{
public:
	CPositivity(void);
	~CPositivity(void);
	void DoIt(MrcUtil::CTomoStack* pTomoStack);
private:
	float mCalcMin(int iFrame);
	void mSetPositivity(int iFrame);
	MrcUtil::CTomoStack* m_pTomoStack;
	float m_fMissingVal;
	float m_fMin;
};

class CFlipInt3D
{
public:
	CFlipInt3D(void);
	~CFlipInt3D(void);
	void DoIt(MrcUtil::CTomoStack* pTomoStack);
};

class GPositivity
{
public:
	GPositivity(void);
	~GPositivity(void);
	void DoIt(MrcUtil::CTomoStack* pTomoStack);
};
}
