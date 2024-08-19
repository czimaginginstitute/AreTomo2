#include "CInput.h"
#include "CProcessThread.h"
#include "Massnorm/CMassNormInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>

static MrcUtil::CTomoStack* s_pTomoStack = 0L;
bool mCheckLoad(void);
bool mCheckSave(char* pcMrcFile);
bool mCheckGPUs(void);
bool mLoadAlignment(void);

int main(int argc, char* argv[])
{
	CInput* pInput = CInput::GetInstance();
	if(argc == 2)
	{	if(strcasecmp(argv[1], "--version") == 0 ||
		   strcasecmp(argv[1], "-v") == 0)
		{	printf("AreTomo2 version 1.1.3\n"
			       "Built on Aug 19 2024\n");
		}
		else if(strcasecmp(argv[1], "--help") == 0)
		{	printf("\nUsage: AreTomo2 Tags\n");
			pInput->ShowTags();
		}
		return 0;
	}	
	//---------------
	cuInit(0);
	pInput->Parse(argc, argv);
	//-----------------
	bool bLoad = mCheckLoad();
	bool bSave = mCheckSave(pInput->m_acOutMrcFile);
	bool bGpu = mCheckGPUs();
	if(!bLoad || !bSave || !bGpu) return 1;
	//-----------------
	Util_Time aTimer;
	aTimer.Measure();
	//-----------------
	CProcessThread aProcessThread;
        aProcessThread.DoIt();
        aProcessThread.WaitForExit(36000000.0f);	
	//-----------------
	float fSecs = aTimer.GetElapsedSeconds();
	printf("Total time: %.2f sec\n\n", fSecs);
	return 0;
}

bool mCheckLoad(void)
{
	CInput* pInput = CInput::GetInstance();
	//-----------------
	Mrc::CLoadMrc aLoadMrc;
	bool bLoad = aLoadMrc.OpenFile(pInput->m_acInMrcFile);
	if(!bLoad)
	{	fprintf(stderr, "Error: unable to open MRC file\n");
		fprintf(stderr, "       %s\n\n", pInput->m_acInMrcFile);
		return false;
	}	
	//-----------------
	int iMode = aLoadMrc.m_pLoadMain->GetMode();
	if(iMode != 1 && iMode != 2 && iMode != 6)
	{	fprintf(stderr, "Error: invalid MRC mode %d\n\n", iMode);
		return false;
	} 	
	//-----------------
	return true;
}

bool mCheckSave(char* pcMrcFile)
{
	CInput* pInput = CInput::GetInstance();
	Mrc::CSaveMrc aSaveMrc;
	bool bSave = aSaveMrc.OpenFile(pcMrcFile);
	remove(pcMrcFile);
	if(bSave) return true;
	//-----------------
	printf("Error: Unable to open output MRC file.\n");
	printf("......%s\n\n", pcMrcFile);
	return false;
}
	
bool mCheckGPUs(void)
{
	CInput* pInput = CInput::GetInstance();
	int* piGpuIDs = new int[pInput->m_iNumGpus];
	if(pInput->m_iNumGpus < 0)
	{	printf("Error: No GPU is specified.\n");
		return false;
	}
	//-----------------
	int iCount = 0;
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	int iGpuId = pInput->m_piGpuIDs[i];
		cudaError_t tErr = cudaSetDevice(iGpuId);
		if(tErr == cudaSuccess)
		{	piGpuIDs[iCount] = iGpuId;
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
	//-----------------
	if(iCount == pInput->m_iNumGpus)
	{	delete[] piGpuIDs;
	}
	else if(iCount < pInput->m_iNumGpus)
	{	if(pInput->m_piGpuIDs != 0L)
		{	delete[] pInput->m_piGpuIDs;
		}
		pInput->m_piGpuIDs = piGpuIDs;
		pInput->m_iNumGpus = iCount;
	}
	if(pInput->m_iNumGpus <= 0) return false;
	//-----------------
	return true;
}

