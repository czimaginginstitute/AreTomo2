#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace Recon;

CDoBaseRecon::CDoBaseRecon(void)
{
	m_gfPadSinogram = 0L;
	m_pfPadSinogram = 0L;
	m_gfVolXZ = 0L;
	m_pfVolXZ = 0L;
	m_iGpuID = -1;
}

CDoBaseRecon::~CDoBaseRecon(void)
{
	this->Clean();
}

void CDoBaseRecon::Clean(void)
{
	if(m_iGpuID < 0) return;
	cudaSetDevice(m_iGpuID);
	if(m_gfPadSinogram != 0L) cudaFree(m_gfPadSinogram);
	if(m_pfPadSinogram != 0L) cudaFreeHost(m_pfPadSinogram);
	if(m_gfVolXZ != 0L) cudaFree(m_gfVolXZ);
	if(m_pfVolXZ != 0L) cudaFreeHost(m_pfVolXZ);
	m_gfPadSinogram = 0L;
	m_pfPadSinogram = 0L;
	m_gfVolXZ = 0L;
	m_pfVolXZ = 0L;
}

void CDoBaseRecon::Run(Util::CNextItem* pNextItem, int iGpuID)
{
	this->Clean();
	//------------
	m_pNextItem = pNextItem;
	m_iGpuID = iGpuID;
	this->Start();
}




	

