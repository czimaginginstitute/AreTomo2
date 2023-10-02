#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <Util/Util_Thread.h>
#include <cufft.h>

namespace DoseWeight 
{

class GDoseWeightImage
{
public:
	GDoseWeightImage(void);
	~GDoseWeightImage(void);
	void Clean(void);
	void BuildWeight(float fPixelSize, float fKv,
	   float* pfImgDose, int* piStkSize, 
	   cudaStream_t stream = 0);
	void DoIt(cufftComplex* gCmpImg, float fDose,
	   cudaStream_t stream = 0);
	float* m_gfWeightSum;
	int m_aiCmpSize[2];
};

class CWeightTomoStack : public Util_Thread
{
public:
	CWeightTomoStack(void);
	virtual ~CWeightTomoStack(void);
	static void DoIt(MrcUtil::CTomoStack* pTomoStack,
	   int* piGpuIDs, int iNumGpus);
	void Clean(void);
	void Run(Util::CNextItem* pNextItem, int iGpuID);
	void ThreadMain(void);
private:
	void mCorrectProj(int iProj);
	void mForwardFFT(int iProj);
	void mInverseFFT(int iProj);
	void mDoseWeight(int iProj);
	cufftComplex* m_gCmpImg;
	int m_aiCmpSize[2];
	Util::CCufft2D m_aForwardFFT;
	Util::CCufft2D m_aInverseFFT;
	Util::CNextItem* m_pNextItem;
	GDoseWeightImage* m_pGDoseWeightImg;
	int m_iGpuID;
};
}
