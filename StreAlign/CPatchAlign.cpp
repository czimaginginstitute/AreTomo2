#include "CAlignInc.h"
#include "../../Util/CUtilInc.h"
#include "../../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace TomoAlign;

CPatchAlign::CPatchAlign(void)
{
	m_pStackShifts = 0L;
}

CPatchAlign::~CPatchAlign(void)
{
	this->Clean();
}

void CPatchAlign::Clean(void)
{
	if(m_pStackShifts != 0L) delete[] m_pStackShifts;
	m_pStackShifts = 0L;
}

void CPatchAlign::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
        CAlignParam* pAlignParam
)
{	this->Clean();
	m_pTomoStack = pTomoStack;
	m_pAlignParam = pAlignParam;
	//--------------------------
	int iPatches = m_pAlignParam->GetPatches();
	m_pStackShifts = new CStackShift[iPatches];
	//-----------------------------------------
	for(int i=0; i<iPatches; i++)
	{	int iLeft = iPatches - 1 - i;
		printf("Align patch %d,  %d left\n\n", i+1, iLeft);
		mAlignPatch(i);
	}
	//---------------------
	float fTiltAxis = m_pTomoStack->GetTiltAxis(0);
	CFitTiltAxis aFitTiltAxis;
	aFitTiltAxis.DoIt
	(  fTiltAxis, 3.0f,
	   m_pStackShifts, m_pAlignParam
	);
	//------------------------------
	CCorrTiltAxis aCorrTiltAxis;
	aCorrTiltAxis.DoIt(m_pTomoStack, &aFitTiltAxis);
}

CStackShift* CPatchAlign::GetPatchShifts(bool bClean)
{
	CStackShift* pPatchShifts = m_pStackShifts;
	if(bClean) m_pStackShifts = 0L;
	return pPatchShifts;
}

void CPatchAlign::mAlignPatch(int iPatch)
{
	int iImgStartX = m_pTomoStack->m_aiStkSize[0] / 20;
	int iImgStartY = m_pTomoStack->m_aiStkSize[1] / 20;
	int iImgSizeX = m_pTomoStack->m_aiStkSize[0] * 9 / 10;
	int iImgSizeY = m_pTomoStack->m_aiStkSize[1] * 9 / 10;
	//----------------------------------------------------
	int* piPatches = m_pAlignParam->m_aiPatches;
	int iX = iPatch % piPatches[0];
	int iY = iPatch / piPatches[0];
	int iSizeX = iImgSizeX / piPatches[0];
	int iSizeY = iImgSizeY / piPatches[1];
	int iCentX = iImgStartX + iX * iSizeX + iSizeX / 2;
	int iCentY = iImgStartX + iY * iSizeY + iSizeY / 2;
	//-------------------------------------------------
	int aiSize[] = {0, 0, m_pTomoStack->m_aiStkSize[2]};
	aiSize[0] = iSizeX / 3 * 4;
	aiSize[1] = iSizeY / 3 * 4;
	//-------------------------
	int aiStart[3] = {0};
	aiStart[0] = iCentX - aiSize[0] / 2;
	aiStart[1] = iCentY - aiSize[1] / 2;
	if(aiStart[0] < 0) aiStart[0] = 0;
	if(aiStart[1] < 0) aiStart[1] = 0;
	if((aiStart[0] + aiSize[0]) > m_pTomoStack->m_aiStkSize[0])
	{	aiStart[0] = m_pTomoStack->m_aiStkSize[0] - aiSize[0];
	}
	if((aiStart[1] + aiSize[1]) > m_pTomoStack->m_aiStkSize[1])
	{	aiStart[1] = m_pTomoStack->m_aiStkSize[1] - aiSize[1];
	}	
	//------------------------------------------------------------
	CStackShift* pPatchShift = m_pStackShifts + iPatch;
	pPatchShift->Setup(m_pTomoStack->m_aiStkSize[2]);
	pPatchShift->SetRegion(aiStart, aiSize);
	//--------------------------------------
	MrcUtil::GExtractPatch aGExtractPatch;
	aGExtractPatch.SetStack(m_pTomoStack);
	MrcUtil::CTomoStack* pPatchStack = aGExtractPatch.DoIt
	(  aiStart, aiSize
	);
	//----------------
	CIterAlign aIterAlign;
	int iXcfBin = aiSize[0] / 512;
	if(iXcfBin < 1) iXcfBin = 1;
	aIterAlign.DoIt(pPatchStack, pPatchShift, m_pAlignParam, iXcfBin);
	//----------------------------------------------------------------
	/* Debugging code */
	char acMrcFile[256];
	CInput* pInput = CInput::GetInstance();
	sprintf(acMrcFile, "%s%d", pInput->m_acOutMrcFile, iPatch);
        MrcUtil::CSaveStack aSaveStack;
        aSaveStack.OpenFile(acMrcFile);
        aSaveStack.DoIt(pPatchStack);
	/* End of debugging */
	//--------------------
	if(pPatchStack != 0L) delete pPatchStack; 
	
	if(strlen(pInput->m_acLogFile) == 0) return;
	char acLogFile[256];
	sprintf
	(  acLogFile, "%s-PatchShift%d.log", 
	   pInput->m_acLogFile, iPatch
	);
	FILE* pLogFile = fopen(acLogFile, "wt");
	pPatchShift->LogShift(pLogFile);
	if(pLogFile != 0L) fclose(pLogFile);
}
