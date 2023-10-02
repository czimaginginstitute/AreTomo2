#include "CFFTBuffer.h"
#include "CInput.h"
#include <memory.h>

CFFTBuffer* CFFTBuffer::m_pInstance = 0L;

CFFTBuffer* CFFTBuffer::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CFFTBuffer;
	return m_pInstance;
}

void CFFTBuffer::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CFFTBuffer::CFFTBuffer(void)
{
	CInput* pInput = CInput::GetInstance();
	m_iNumGpus = pInput->m_iNumGpus;
	if(m_iNumGpus <= 0) return;
	//-------------------------
	m_gForward2Ds = new CuUtilFFT::GFFT2D[m_iNumGpus * 2];
	m_gInverse2Ds = m_gForward2Ds + m_iNumGpus;
	//-----------------------------------------
	m_gForward1Ds = new CuUtilFFT::GFFT1D[m_iNumGpus * 2];
	m_gInverse1Ds = m_gForward1Ds + m_iNumGpus;
}

CFFTBuffer::~CFFTBuffer(void)
{
	if(m_iNumGpus <= 0) return;
	delete[] m_gForward2Ds;
	delete[] m_gForward1Ds;
	m_iNumGpus = 0;
}

CuUtilFFT::GFFT2D* CFFTBuffer::GetForward2D(int iNthGpu)
{
	if(iNthGpu < 0 || iNthGpu >= m_iNumGpus) return 0L;
	return &m_gForward2Ds[iNthGpu];
}

CuUtilFFT::GFFT2D* CFFTBuffer::GetInverse2D(int iNthGpu)
{
	if(iNthGpu < 0 || iNthGpu >= m_iNumGpus) return 0L;
	return &m_gInverse2Ds[iNthGpu];
}

CuUtilFFT::GFFT1D* CFFTBuffer::GetForward1D(int iNthGpu)
{
	if(iNthGpu < 0 || iNthGpu >= m_iNumGpus) return 0L;
	return &m_gForward1Ds[iNthGpu];
}

CuUtilFFT::GFFT1D* CFFTBuffer::GetInverse1D(int iNthGpu)
{
	if(iNthGpu < 0 || iNthGpu >= m_iNumGpus) return 0L;
	return &m_gInverse1Ds[iNthGpu];
}

