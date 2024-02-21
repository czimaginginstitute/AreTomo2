#pragma once
#include "Util/CUtilInc.h"

class CFFTBuffer
{
public:
	static CFFTBuffer* GetInstance(void);
	static void DeleteInstance(void);
	~CFFTBuffer(void);
	Util::GFFT2D* GetForward2D(int iNthGpu);
	Util::GFFT2D* GetInverse2D(int iNthGpu);
	Util::GFFT1D* GetForward1D(int iNthGpu);
	Util::GFFT1D* GetInverse1D(int iNthGpu);	
private:
	CFFTBuffer(void);
	int m_iNumGpus;
	Util::GFFT2D* m_gForward2Ds;
	Util::GFFT2D* m_gInverse2Ds;
	Util::GFFT1D* m_gForward1Ds;
	Util::GFFT1D* m_gInverse1Ds;
        static CFFTBuffer* m_pInstance;
};
