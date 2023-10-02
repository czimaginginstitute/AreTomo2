#include "CMainInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace TomoRalign;

CProcessThread::CProcessThread(void)
{
	m_pTomoStack = 0L;
}

CProcessThread::~CProcessThread(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
}

bool CProcessThread::DoIt
(	MrcUtil::CTomoStack* pTomoStack
)
{	bool bExit = this->WaitForExit(10000.0f);
	if(!bExit) return false;
	//----------------------
	m_pTomoStack = pTomoStack;
	this->Start();
	return true;
}

void CProcessThread::ThreadMain(void)
{
	printf("\nReconstruction thread has been started.\n\n");
	//------------------------------------------------------
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIds[0]);
	m_pCalcProjs = new Recon::CCalcProjs;
	mBin2x();
	mDoIt();
	mSaveTomoMrc();
	if(m_pCalcProjs != 0L) delete m_pCalcProjs;
	//---------------------------------------
	printf("Process thread exits.\n\n");
}

void CProcessThread::mBin2x(void)
{
	int aiBinning[] = {2, 2};
	int aiBinSize[] = {0, 0};
	Util::GBinImage2D aGBinImg2D;
	aGBinImg2D.Setup(m_pTomoStack->m_aiProjSize, aiBinning);
	aGBinImg2D.GetBinSize(aiBinSize);
	//-------------------------------
	int iBinSize = aiBinSize[0] * aiBinSize[1];
	size_t tBytes = sizeof(float) * iBinSize;
	float* gfBinProj = 0L;
	cudaMalloc(&gfBinProj, tBytes);
	//-----------------------------
	bool bGpu = true;
	bool bClean = true;
	cudaMemcpyKind aD2H = cudaMemcpyDeviceToHost;
	//-------------------------------------------
	for(int i=0; i<m_pTomoStack->m_iNumProjs; i++)
	{	printf("...... Bin proj %4d\n", i+1);
		float* pfProj = m_pTomoStack->GetProj(i, bClean);
		aGBinImg2D.DoIt(pfProj, !bGpu, gfBinProj);
		if(pfProj != 0L) delete[] pfProj;
		pfProj = new float[iBinSize];
		cudaMemcpy(pfProj, gfBinProj, tBytes, aD2H);
		m_pTomoStack->SetProj(i, pfProj);
	}
	m_pTomoStack->m_aiProjSize[0] = aiBinSize[0];
	m_pTomoStack->m_aiProjSize[1] = aiBinSize[1];
	//-------------------------------------------
	if(gfBinProj != 0L) cudaFree(gfBinProj);
}

void CProcessThread::mDoIt(void)
{
	printf("Start forward projection......\n");
	CInput* pInput = CInput::GetInstance();
	pInput->m_aiVolSize[0] = m_pTomoStack->m_aiProjSize[0];
	pInput->m_aiVolSize[1] = m_pTomoStack->m_aiProjSize[1];
	if(pInput->m_aiVolSize[2] <= 0)
	{	pInput->m_aiVolSize[2] = 256;
	}
	//-----------------------------------
	bool bExclusive = true;
	float* pfAngles = m_pTomoStack->GetTiltAngles();
	m_pCalcProjs->SetProjs
	(  m_pTomoStack->m_ppfProjs,
	   m_pTomoStack->m_aiProjSize,
	   pfAngles, m_pTomoStack->m_iNumProjs
	);
	m_pCalcProjs->DoIt
	(  pInput->m_aiVolSize[2],
	   bExclusive 
	);	   
	if(pfAngles != 0L) delete[] pfAngles;
	//-----------------------------------
	printf("Forward projection: done.\n\n");
}

void CProcessThread::mSaveTomoMrc(void)
{
	printf("Save forward projections into MRC file......\n");
	Mrc::CSaveMrc aSaveMrc;
	CInput* pInput = CInput::GetInstance();
	aSaveMrc.OpenFile(pInput->m_acOutMrcFile);
	aSaveMrc.SetMode(Mrc::eMrcFloat);
	aSaveMrc.SetImgSize
	(  m_pCalcProjs->m_aiProjSize,
	   m_pCalcProjs->m_iNumProjs
	);
	aSaveMrc.SetExtHeader(0, 0, m_pCalcProjs->m_iNumProjs, 0);
	aSaveMrc.SetPixelSize(m_pTomoStack->m_fPixelSize);
	aSaveMrc.m_pSaveMain->DoIt();
	//---------------------------
	bool bClean = true;
	for(int i=0; i<m_pCalcProjs->m_iNumProjs; i++)
	{	float* pfReproj = m_pCalcProjs->GetReproj(i, bClean);
		aSaveMrc.m_pSaveImg->DoIt(i, pfReproj);
		if(pfReproj != 0L) delete[] pfReproj;
	}
	printf("Save volume: done\n\n");
}

