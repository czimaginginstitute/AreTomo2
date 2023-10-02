#include "CMainInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace TomoRalign;

bool mCheckLoad(void);
bool mCheckSave(char* pcMrcFile);
bool mCheckGPUs(void);

int main(int argc, char* argv[])
{
	//cuInit(0);
	printf("\nUsage: TomoForProj Tags\n");
	CInput* pInput = CInput::GetInstance();
	pInput->ShowTags();
	pInput->Parse(argc, argv);
	//------------------------
	bool bLoad = mCheckLoad();
	bool bSave = mCheckSave(pInput->m_acOutMrcFile);
	bool bGpu = mCheckGPUs();
	if(!bLoad || !bSave || !bGpu) return 1;
	//-------------------------------------
	Util_Time aTimer;
	aTimer.Measure();
	CMain aMain;
	aMain.DoIt();
	float fSecs = aTimer.GetElapsedSeconds();
	printf("Total time: %f sec\n", fSecs);
	return 0;
}

bool mCheckLoad(void)
{
	CInput* pInput = CInput::GetInstance();
	//-------------------------------------
	Mrc::CLoadMrc aLoadMrc;
	bool bLoad = aLoadMrc.OpenFile(pInput->m_acInMrcFile);
	if(bLoad) return true;
	printf("Error: Unable to open input MRC file.\n");
	printf("       %s\n\n", pInput->m_acInMrcFile);
	return false;
}

bool mCheckSave(char* pcMrcFile)
{
	CInput* pInput = CInput::GetInstance();
	Mrc::CSaveMrc aSaveMrc;
	bool bSave = aSaveMrc.OpenFile(pcMrcFile);
	remove(pcMrcFile);
	if(bSave) return true;
	//--------------------
	printf("Error: Unable to open output MRC file.\n");
	printf("......%s\n\n", pcMrcFile);
	return false;
}
	
bool mCheckGPUs(void)
{
	CInput* pInput = CInput::GetInstance();
	int* piGpuIds = new int[pInput->m_iNumGpus];
	if(pInput->m_iNumGpus < 0)
	{	printf("Error: No GPU is specified.\n");
		return false;
	}
	//-------------------
	int iCount = 0;
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	int iGpuId = pInput->m_piGpuIds[i];
		cudaError_t tErr = cudaSetDevice(iGpuId);
		if(tErr == cudaSuccess)
		{	piGpuIds[iCount] = iGpuId;
			iCount++;
			continue;
		}
		printf("Error: GPU %d is invalid, skip\n", iGpuId);
		if(tErr == cudaErrorInvalidDevice)
		{	printf("...... Indvalid device.\n\n");
		}
		else if(tErr == cudaErrorDeviceAlreadyInUse)
		{	printf("...... Device already in use.\n\n");
		}
	}
	//----------------------------------------------------------
	if(iCount == pInput->m_iNumGpus)
	{	delete[] piGpuIds;
	}
	else if(iCount < pInput->m_iNumGpus)
	{	if(pInput->m_piGpuIds != 0L)
		{	delete[] pInput->m_piGpuIds;
		}
		pInput->m_piGpuIds = piGpuIds;
		pInput->m_iNumGpus = iCount;
	}
	if(pInput->m_iNumGpus <= 0) return false;
	//---------------------------------------
	cudaSetDevice(pInput->m_piGpuIds[0]);
	for(int i=1; i<pInput->m_iNumGpus; i++)
	{	cudaDeviceEnablePeerAccess(pInput->m_piGpuIds[i], 0);
	}
	for(int i=1; i<pInput->m_iNumGpus; i++)
	{	cudaSetDevice(pInput->m_piGpuIds[i]);
		cudaDeviceEnablePeerAccess(pInput->m_piGpuIds[0], 0);
	}
	cudaSetDevice(pInput->m_piGpuIds[0]);
	return true;
}
