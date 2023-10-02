#pragma once
#include <CuUtilFFT/GFFT2D.h>
#include <CuUtilFFT/GFFT1D.h>

class CFFTBuffer
{
public:
	static CFFTBuffer* GetInstance(void);
	static void DeleteInstance(void);
	~CFFTBuffer(void);
	CuUtilFFT::GFFT2D* GetForward2D(int iNthGpu);
	CuUtilFFT::GFFT2D* GetInverse2D(int iNthGpu);
	CuUtilFFT::GFFT1D* GetForward1D(int iNthGpu);
	CuUtilFFT::GFFT1D* GetInverse1D(int iNthGpu);	
private:
	CFFTBuffer(void);
	int m_iNumGpus;
	CuUtilFFT::GFFT2D* m_gForward2Ds;
	CuUtilFFT::GFFT2D* m_gInverse2Ds;
	CuUtilFFT::GFFT1D* m_gForward1Ds;
	CuUtilFFT::GFFT1D* m_gInverse1Ds;
        static CFFTBuffer* m_pInstance;
};
