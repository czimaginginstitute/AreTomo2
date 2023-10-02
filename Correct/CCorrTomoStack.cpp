#include "CCorrectInc.h"
#include "../Util/CUtilInc.h"
#include "../CInput.h"
#include "../PatchAlign/CPatchAlignInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>

using namespace Correct;

static float* sAllocGPU(int* piSize, bool bPad)
{
	float* gfBuf = 0L;
	int iSizeX = bPad ? (piSize[0] / 2 + 1) * 2 : piSize[0];
	size_t tBytes = sizeof(float) * iSizeX * piSize[1];
	cudaMalloc(&gfBuf, tBytes);
	return gfBuf;
}

static void sCalcPadSize(int* piSize, int* piPadSize)
{
	piPadSize[0] = (piSize[0] / 2 + 1) * 2;
	piPadSize[1] = piSize[1];
}

static void sCalcCmpSize(int* piSize, int* piCmpSize)
{
	piCmpSize[0] = piSize[0] / 2 + 1;
	piCmpSize[1] = piSize[1];
}

CCorrTomoStack::CCorrTomoStack(void)
{
	m_gfRawProj = 0L;
	m_gfCorrProj = 0L;
	m_gfBinProj = 0L;
	m_gfLocalParam = 0L;
	m_pOutStack = 0L;
	m_pGRWeight = 0L;
	m_pCorrInt = 0L;
	m_iGpuID = -1;
}

CCorrTomoStack::~CCorrTomoStack(void)
{
	this->Clean();
}

void CCorrTomoStack::Clean(void)
{
	if(m_iGpuID < 0) return;
	cudaSetDevice(m_iGpuID);
	//----------------------
	if(m_gfRawProj != 0L) cudaFree(m_gfRawProj);
	if(m_gfCorrProj != 0L) cudaFree(m_gfCorrProj);
	if(m_gfBinProj != 0L) cudaFree(m_gfBinProj);
	if(m_gfLocalParam != 0L) cudaFree(m_gfLocalParam);
	if(m_pOutStack != 0L) delete m_pOutStack;
	if(m_pGRWeight != 0L) delete (Recon::GRWeight*)m_pGRWeight;
	if(m_pCorrInt != 0L) delete m_pCorrInt;
	m_gfRawProj = 0L;
	m_gfCorrProj = 0L;
	m_gfBinProj = 0L;
	m_gfLocalParam = 0L;
	m_pOutStack = 0L;
	m_pGRWeight = 0L;
	m_pCorrInt = 0L;
}

void CCorrTomoStack::GetBinning(float* pfBinning)
{
	pfBinning[0] = m_afBinning[0];
	pfBinning[1] = m_afBinning[1];
}

void CCorrTomoStack::Set0(int iGpuID)
{
	this->Clean();
	m_iGpuID = iGpuID;
}

//-----------------------------------------------------------------------------
// Note: In case of shift only, 0 must be passed in for fTiltAxis
//-----------------------------------------------------------------------------
void CCorrTomoStack::Set1(int* piStkSize, int iNumPatches, float fTiltAxis)
{
	if(m_iGpuID < 0) return;
	cudaSetDevice(m_iGpuID);
	//----------------------
	memcpy(m_aiStkSize, piStkSize, sizeof(int) * 3);
	CCorrectUtil::CalcAlignedSize(m_aiStkSize, fTiltAxis, m_aiAlnSize);
	m_aiAlnSize[2] = m_aiStkSize[2];
	//------------------------------
	bool bPad = true, bPadded = true;
	m_gfRawProj = sAllocGPU(m_aiStkSize, !bPad);
	//------------------------------------------
	int* piCorrSize = m_aiAlnSize;
	if(m_aiAlnSize[1] < m_aiStkSize[1]) piCorrSize = m_aiStkSize;
	m_gfCorrProj = sAllocGPU(piCorrSize, bPad);
	//-----------------------------------------
        int aiAlnPadSize[2] = {0};
        sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
        m_aGCorrPatchShift.SetSizes(m_aiStkSize, !bPadded,
           aiAlnPadSize, bPadded, iNumPatches);
	//-------------------------------------	
	if(iNumPatches <= 0) return;
	int iBytes = iNumPatches * 5 * sizeof(float);
	cudaMalloc(&m_gfLocalParam, iBytes);
}

void CCorrTomoStack::Set2(float fOutBin, bool bFourierCrop, bool bRandFill)
{
	if(m_iGpuID < 0) return;
	cudaSetDevice(m_iGpuID);
	//----------------------
	m_fOutBin = fOutBin;
	m_bFourierCrop = bFourierCrop;
	m_bRandomFill = m_bFourierCrop ? true : bRandFill;
	//------------------------------------------------
	CCorrectUtil::CalcBinnedSize(m_aiAlnSize, m_fOutBin,
	   m_bFourierCrop, m_aiBinnedSize);
	m_aiBinnedSize[2] = m_aiStkSize[2];
	//---------------------------------
	m_afBinning[0] = m_aiAlnSize[0] / (float)m_aiBinnedSize[0];
	m_afBinning[1] = m_aiAlnSize[1] / (float)m_aiBinnedSize[1];
	//---------------------------------------------------------
	bool bPad = true;
	m_gfBinProj = sAllocGPU(m_aiBinnedSize, bPad);
	//--------------------------------------------
	if(m_pOutStack != 0L) delete m_pOutStack;
	m_pOutStack = new MrcUtil::CTomoStack;
	m_pOutStack->Create(m_aiBinnedSize, true);
	//----------------------------------------
	if(m_bFourierCrop)
	{	m_aFFTCropImg.Setup(0, m_aiAlnSize, m_fOutBin);
	}
	else
	{	bool bPadded = true;
		int aiAlnPadSize[2], aiBinPadSize[2];
		sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
		sCalcPadSize(m_aiBinnedSize, aiBinPadSize);
		m_aGBinImg2D.SetupSizes(aiAlnPadSize, bPadded,
		   aiBinPadSize, bPadded);
	}
}

void CCorrTomoStack::Set3(bool bShiftOnly, bool bCorrInt, bool bRWeight)
{	
	if(m_iGpuID < 0) return;
	cudaSetDevice(m_iGpuID);
	//----------------------
	m_bShiftOnly = bShiftOnly;
	//------------------------
	int aiBinPadSize[2], aiAlnPadSize[2];
	sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
	sCalcPadSize(m_aiBinnedSize, aiBinPadSize);
	if(bCorrInt)
	{	m_pCorrInt = new CCorrLinearInterp;
		m_pCorrInt->Setup(m_aiStkSize);
	}
	//-------------------------------------
	if(bRWeight)
	{	Recon::GRWeight* pGRWeight = new Recon::GRWeight;
		pGRWeight->SetSize(aiBinPadSize[0], aiBinPadSize[1]);
		m_pGRWeight = pGRWeight;
	}
}

MrcUtil::CTomoStack* CCorrTomoStack::GetCorrectedStack(bool bClean)
{
	MrcUtil::CTomoStack* pRetStack = m_pOutStack;
	if(bClean) m_pOutStack = 0L;
	return pRetStack;
}

void CCorrTomoStack::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	MrcUtil::CLocalAlignParam* pLocalParam
)
{	if(m_iGpuID < 0) return;
	cudaSetDevice(m_iGpuID);
	//----------------------
	m_pTomoStack = pTomoStack;
	m_pFullParam = pAlignParam;
	m_pLocalParam = pLocalParam;
	//--------------------------
	for(int i=0; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	mCorrectProj(i);
	}
}

void CCorrTomoStack::mCorrectProj(int iProj)
{
	float afShift[2] = {0.0f};
	m_pFullParam->GetShift(iProj, afShift);
	//--------------------------------------
	float fTiltAxis = m_pFullParam->GetTiltAxis(iProj);
	if(m_bShiftOnly) fTiltAxis = 0.0f;
	//--------------------------------
	float* pfProj = m_pTomoStack->GetFrame(iProj);
	size_t tBytes = sizeof(float) * m_pTomoStack->GetPixels();
	cudaMemcpy(m_gfRawProj, pfProj, tBytes, cudaMemcpyDefault);
	//---------------------------------------------------------
	if(!m_bShiftOnly && m_pCorrInt != 0L)
	{	m_pCorrInt->DoIt(m_gfRawProj, m_gfCorrProj);
	}
	//--------------------------------------------------
	if(m_pLocalParam != 0L) 
	{	m_pLocalParam->GetParam(iProj, m_gfLocalParam);
	}
	//-----------------------------------------------------
	m_aGCorrPatchShift.DoIt(m_gfRawProj, afShift, fTiltAxis, 
	   m_gfLocalParam, m_bRandomFill, m_gfCorrProj);
	//----------------------------------------------
	if(m_bFourierCrop)
	{	m_aFFTCropImg.DoPad(m_gfCorrProj, m_gfBinProj);
	}
	else 
	{	m_aGBinImg2D.DoIt(m_gfCorrProj, m_gfBinProj);
	}
	//---------------------------------------------------
	if(m_pGRWeight != 0L)
	{	Recon::GRWeight* pGRWeight = (Recon::GRWeight*)m_pGRWeight;
		pGRWeight->DoIt(m_gfBinProj);
	}
	//-----------------------------------
        int aiPadSize[] = {0, m_pOutStack->m_aiStkSize[1]};
	aiPadSize[0] = (m_pOutStack->m_aiStkSize[0] / 2 + 1) * 2;
	float* pfProjOut = m_pOutStack->GetFrame(iProj);
        Util::CPad2D pad2D;
	pad2D.Unpad(m_gfBinProj, aiPadSize, pfProjOut);
}

