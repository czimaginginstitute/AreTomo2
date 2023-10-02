#include "CProcessThread.h"
#include "CInput.h"
#include "CFFTBuffer.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include "FindTilts/CFindTiltsInc.h"
#include "StreAlign/CStreAlignInc.h"
#include "ProjAlign/CProjAlignInc.h"
#include "PatchAlign/CPatchAlignInc.h"
#include "CommonLine/CCommonLineInc.h"
#include "Massnorm/CMassNormInc.h"
#include "TiltOffset/CTiltOffsetInc.h"
#include "Recon/CReconInc.h"
#include "DoseWeight/CDoseWeightInc.h"
#include "FindCtf/CFindCtfInc.h"
#include "ImodUtil/CImodUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Util/Util_Time.h>

namespace SA = StreAlign;
namespace PA = ProjAlign;

CProcessThread::CProcessThread(void)
{
	m_pTomoStack = 0L;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
	m_pCorrTomoStack = 0L;
}

CProcessThread::~CProcessThread(void)
{
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
}

bool CProcessThread::DoIt(MrcUtil::CTomoStack* pTomoStack)
{	
	bool bExit = this->WaitForExit(10000000.0f);
	if(!bExit) return false;
	//----------------------
	m_pTomoStack = pTomoStack;
	this->Start();
	return true;
}

void CProcessThread::ThreadMain(void)
{
	printf("\nProcess thread has been started.\n\n");
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[0]);
	//-----------------------------------
	MrcUtil::CLoadAlignment* pLoadAlign = 
	   MrcUtil::CLoadAlignment::GetInstance();
	m_pAlignParam = pLoadAlign->GetAlignParam(true);
	m_pLocalParam = pLoadAlign->GetLocalParam(true);
	//----------------------------------------------
	PA::CRemoveSpikes::DoIt(m_pTomoStack, pInput->m_piGpuIDs,
	   pInput->m_iNumGpus);
	mSetPositivity();	
	//---------------
	mFindCtf();
	mAlign();
	mDoseWeight();
	mSetPositivity();
	mSaveAlignment();
	mCorrectStack();
	mRecon();
	mCropVol();
	mFlipInt();
        mSaveCentralSlices();
        mFlipVol();
	mSaveStack();
	//-----------
	if(m_pTomoStack != 0L) delete m_pTomoStack;
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	m_pTomoStack = 0L;
	m_pAlignParam = 0L;
	//-----------------
	printf("Process thread exits.\n\n");
}

void CProcessThread::mFindCtf(void)
{
	FindCtf::CFindCtfMain aFindCtfMain;
	if(!aFindCtfMain.CheckInput()) return;
	//------------------------------------
	aFindCtfMain.DoIt(m_pTomoStack, m_pAlignParam);
}

void CProcessThread::mAlign(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iAlign == 0) return;
	//-------------------------------------
	PatchAlign::CRoiTargets* pRoiTargets = 
	   PatchAlign::CRoiTargets::GetInstance();
	pRoiTargets->LoadRoiFile();
	pRoiTargets->SetTargetImage(m_pAlignParam);
	//-----------------------------------------	
	MassNorm::CLinearNorm aLinearNorm;
	aLinearNorm.DoIt(m_pTomoStack, m_pAlignParam);
	//--------------------------------------------
	MrcUtil::CLoadAlignment* pLoadAlign = 
	   MrcUtil::CLoadAlignment::GetInstance();
	if(pLoadAlign->m_bFromAlnFile) return;
	//------------------------------------
	m_fRotScore = 0.0f;
	mCoarseAlign();
	//------------------------------------
	m_pAlignParam->ResetShift();
	ProjAlign::CParam* pParam = ProjAlign::CParam::GetInstance();
	pParam->m_fXcfSize = 2048.0f;
	mProjAlign();
	//-----------
	if(pInput->m_afTiltAxis[1] >= 0)
	{	float fRange = (pInput->m_afTiltAxis[0] == 0) ? 20.0f : 6.0f;
		int iIters = (pInput->m_afTiltAxis[0] == 0) ? 4 : 2;
		for(int i=1; i<=iIters; i++) 
		{	mRotAlign(fRange/i, 100);
			if(i == 1) mProjAlign();
		}
		mProjAlign();
	}
	//------------------------------------------------------------------
	pRoiTargets->MapToUntiltImage(m_pAlignParam, m_pTomoStack);
	mPatchAlign();
	//------------
	if(pInput->m_afTiltCor[0] == 0) 
	{	m_pAlignParam->AddTiltOffset(-m_fTiltOffset);
	}
	else
	{	MrcUtil::CAcqSequence* pAcqSequence = 
		   MrcUtil::CAcqSequence::GetInstance();
		pAcqSequence->AddTiltOffset(m_fTiltOffset);
		//-----------------------------------------
		MrcUtil::CDarkFrames* pDarkFrames = 
		   MrcUtil::CDarkFrames::GetInstance();
		pDarkFrames->AddTiltOffset(m_fTiltOffset);
	}
	CFFTBuffer::DeleteInstance();
}

void CProcessThread::mCoarseAlign(void)
{
	SA::CStreAlignMain streAlignMain;
	streAlignMain.Setup(m_pTomoStack, m_pAlignParam);
	//-----------------------------------------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_afTiltAxis[0] == 0)
	{	for(int i=1; i<=3; i++)
		{	streAlignMain.DoIt(m_pTomoStack, m_pAlignParam);
			mRotAlign(180.0f / i, 100);
		}
		mFindTiltOffset();
		return;
	}
	//-------------
	m_pAlignParam->SetTiltAxisAll(pInput->m_afTiltAxis[0]);
	if(pInput->m_afTiltAxis[1] < 0)
	{	streAlignMain.DoIt(m_pTomoStack, m_pAlignParam);
		streAlignMain.DoIt(m_pTomoStack, m_pAlignParam);
	}
	else
	{	for(int i=1; i<=2; i++)
		{	streAlignMain.DoIt(m_pTomoStack, m_pAlignParam);
			mRotAlign(10.0f / i, 100);
		}
	}	
        mFindTiltOffset();
}


void CProcessThread::mStretchAlign(void)
{
	SA::CStreAlignMain streAlignMain;
	streAlignMain.DoIt(m_pTomoStack, m_pAlignParam);
}

void CProcessThread::mProjAlign(void)
{
	CInput* pInput = CInput::GetInstance();
	PA::CParam* pParam = PA::CParam::GetInstance();
	pParam->m_iVolZ = pInput->m_iAlignZ;
        pParam->m_afMaskSize[0] = 0.7f;
        pParam->m_afMaskSize[1] = 0.7f;
	//-----------------------------
	PA::CProjAlignMain aProjAlign;
	aProjAlign.Setup(m_pTomoStack, m_pAlignParam, 
	   pInput->m_afBFactor[0], 0);
	float fLastErr = aProjAlign.DoIt(m_pTomoStack, m_pAlignParam);
	MrcUtil::CAlignParam* pLastParam = m_pAlignParam->GetCopy();
	//----------------------------------------------------------
	int iIterations = 10;
	int iLastIter = iIterations - 1;
	pParam->m_afMaskSize[0] = 0.55f;
	pParam->m_afMaskSize[1] = 0.55f;
	//------------------------------
	for(int i=1; i<iIterations; i++)
	{	float fErr = aProjAlign.DoIt(m_pTomoStack, m_pAlignParam);
		if(fErr < 2.0f) break;
		//--------------------
		if(fErr <= fLastErr)
		{	fLastErr = fErr;
			pLastParam->Set(m_pAlignParam);
		}
		else
		{	m_pAlignParam->Set(pLastParam);
			break;
		}
	}
	delete pLastParam;
}

void CProcessThread::mRotAlign(float fAngRange, int iNumSteps)
{
	CInput* pInput = CInput::GetInstance();
	CommonLine::CCommonLineMain clMain;
	clMain.DoInitial(m_pTomoStack, m_pAlignParam, fAngRange, iNumSteps);
}

void CProcessThread::mRotAlign(void)
{
	CInput* pInput = CInput::GetInstance();
        CommonLine::CCommonLineMain clMain;
        m_fRotScore = clMain.DoRefine(m_pTomoStack, m_pAlignParam);
	printf("Rotation align score: %f\n\n", m_fRotScore);
}

void CProcessThread::mFindTiltOffset(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_afTiltCor[0] < 0) return;
	//------------------------------------
	if(fabs(pInput->m_afTiltCor[1]) > 0.1)
        {       m_fTiltOffset = pInput->m_afTiltCor[1];
                m_pAlignParam->AddTiltOffset(m_fTiltOffset);
		return;
        }
	//-------------
	TiltOffset::CTiltOffsetMain aTiltOffsetMain;
	aTiltOffsetMain.Setup(m_pTomoStack->m_aiStkSize, 4, 0);
	float fTiltOffset = aTiltOffsetMain.DoIt(m_pTomoStack, m_pAlignParam);
	m_pAlignParam->AddTiltOffset(fTiltOffset);
	m_fTiltOffset += fTiltOffset;
}

void CProcessThread::mPatchAlign(void)
{
	PatchAlign::CPatchTargets* pPatchTargets = 
	   PatchAlign::CPatchTargets::GetInstance();
	pPatchTargets->DetectTargets(m_pTomoStack, m_pAlignParam);
	if(pPatchTargets->m_iNumTgts < 4) return;
	if(m_pLocalParam != 0L) return;
	m_pLocalParam = PatchAlign::CPatchAlignMain::DoIt(m_pTomoStack, 
	   m_pAlignParam, m_fTiltOffset);
}

void CProcessThread::mDoseWeight(void)
{
	CInput* pInput = CInput::GetInstance();
	DoseWeight::CWeightTomoStack::DoIt(m_pTomoStack, pInput->m_piGpuIDs,
	   pInput->m_iNumGpus);
}

void CProcessThread::mSetPositivity(void)
{
	MassNorm::GPositivity aGPositivity;
	aGPositivity.DoIt(m_pTomoStack);
}

void CProcessThread::mCorrectStack(void)
{
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = new Correct::CCorrTomoStack;
	//---------------------------------------------
	mCorrectForImod();
	//----------------
	CInput* pInput = CInput::GetInstance();
	bool bShiftOnly = true, bRandFill = true; 
	bool bFFTCrop = true, bRWeight = true;
	bool bCorrInterp = (pInput->m_iIntpCor == 0) ? false : true;
	//----------------------------------------------------------
	int iNumPatches = (m_pLocalParam == 0L) ? 0 :
	   m_pLocalParam->m_iNumPatches;
	float fTiltAxis = m_pAlignParam->GetTiltAxis(
	   m_pAlignParam->m_iNumFrames / 2);
	//----------------------------------
	m_pCorrTomoStack->Set0(pInput->m_piGpuIDs[0]);
	m_pCorrTomoStack->Set1(m_pTomoStack->m_aiStkSize, 
	   iNumPatches, fTiltAxis);
	m_pCorrTomoStack->Set2(pInput->m_fOutBin, bFFTCrop, bRandFill);
	m_pCorrTomoStack->Set3(!bShiftOnly, bCorrInterp, !bRWeight);
	m_pCorrTomoStack->DoIt(m_pTomoStack, m_pAlignParam, m_pLocalParam);
	//-----------------------------------------------------------------
	delete m_pTomoStack;
	m_pTomoStack = m_pCorrTomoStack->GetCorrectedStack(true);
}

void CProcessThread::mCorrectForImod(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iOutImod == 0) return;
	//---------------------------------
	ImodUtil::CImodUtil* pImodUtil = 0L;
	pImodUtil = ImodUtil::CImodUtil::GetInstance();
	pImodUtil->CreateFolder();
	//------------------------
	if(pInput->m_iOutImod == 1) // for Relion 4
	{	pImodUtil->SaveTiltSeries(0L, m_pAlignParam,
		   pInput->m_fPixelSize);
		return;
	}
	else if(pInput->m_iOutImod == 2) // for warp
	{	pImodUtil->SaveTiltSeries(m_pTomoStack, m_pAlignParam,
		   pInput->m_fPixelSize);
		return;
	}
	//---------------------------------------
	// for using aligned tilt series as input
	//---------------------------------------
	bool bShiftOnly = true, bRandomFill = true;
	bool bFourierCrop = true, bRWeight = true;
	int iNumPatches = (m_pLocalParam == 0L) ? 0 :
	   m_pLocalParam->m_iNumPatches;
	float fTiltAxis = m_pAlignParam->GetTiltAxis(
	   m_pAlignParam->m_iNumFrames / 2);
	m_pCorrTomoStack->Set0(pInput->m_piGpuIDs[0]);
	m_pCorrTomoStack->Set1(m_pTomoStack->m_aiStkSize, 
	   iNumPatches, fTiltAxis);
	m_pCorrTomoStack->Set2(1.0f, bFourierCrop, bRandomFill);
	m_pCorrTomoStack->Set3(!bShiftOnly, true, !bRWeight);
	m_pCorrTomoStack->DoIt(m_pTomoStack, m_pAlignParam, m_pLocalParam);
	MrcUtil::CTomoStack* pCorStack = 0L;
	pCorStack = m_pCorrTomoStack->GetCorrectedStack(true);
	//----------------------------------------------------
	pImodUtil->SaveTiltSeries(pCorStack, m_pAlignParam,
	   pInput->m_fPixelSize);
	delete pCorStack;
}

void CProcessThread::mRecon(void)
{
	CInput* pInput = CInput::GetInstance();
	int iVolZ = (int)(pInput->m_iVolZ / pInput->m_fOutBin) / 2 * 2;
	if(iVolZ <= 16) return;
	//---------------------
	if(pInput->m_iWbp != 0) mWbpRecon(iVolZ);
	else mSartRecon(iVolZ);
}

void CProcessThread::mCropVol(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iVolZ == 0) return;
	if(pInput->m_aiCropVol[0] < 10) return;
	if(pInput->m_aiCropVol[1] < 10) return;
	if(m_pLocalParam == 0L) return;
	//-----------------------------
	MrcUtil::CCropVolume aCropVolume;
	MrcUtil::CTomoStack* pCroppedVol = aCropVolume.DoIt(m_pTomoStack,
	   pInput->m_fOutBin, m_pAlignParam, m_pLocalParam,
	   pInput->m_aiCropVol);
	delete m_pTomoStack;
	m_pTomoStack = pCroppedVol;
}  

void CProcessThread::mSartRecon(int iVolZ)
{
	CInput* pInput = CInput::GetInstance();
	int iStartTilt = m_pAlignParam->GetFrameIdxFromTilt
	( pInput->m_afReconRange[0]
	);
	int iEndTilt = m_pAlignParam->GetFrameIdxFromTilt
	( pInput->m_afReconRange[1]
	);
	if(iStartTilt == iEndTilt) return;
	//--------------------------------
	int iNumTilts = iEndTilt - iStartTilt + 1;
	int iNumSubsets = iNumTilts / pInput->m_aiSartParam[1];
	if(iNumSubsets < 1) iNumSubsets = 1;
	//----------------------------------
	printf("Start SART reconstruction...\n");
	Util_Time aTimer;
	aTimer.Measure();
	MrcUtil::CTomoStack* pVolStack = Recon::CDoSartRecon::DoIt
	( m_pTomoStack, m_pAlignParam,
	  iStartTilt, iNumTilts,
	  iVolZ, pInput->m_aiSartParam[0], iNumSubsets,
	  pInput->m_piGpuIDs, pInput->m_iNumGpus
	);
	printf("SART Recon: %.2f sec\n\n", aTimer.GetElapsedSeconds());
	delete m_pTomoStack;
	m_pTomoStack = pVolStack;
}

void CProcessThread::mWbpRecon(int iVolZ)
{
	printf("Start WBP reconstruction...\n");
	CInput* pInput = CInput::GetInstance();
	float fRFactor = 6.0f - 4.0f * (pInput->m_fOutBin - 1) / 7.0f;
	if(pInput->m_iWbp == 2) fRFactor = 2.0f;
	//--------------------------------------
	Util_Time aTimer;
	aTimer.Measure();
	MrcUtil::CTomoStack* pVolStack = Recon::CDoWbpRecon::DoIt
	( m_pTomoStack, m_pAlignParam, iVolZ, fRFactor,
	  pInput->m_piGpuIDs, pInput->m_iNumGpus
	);
	printf("WBP Recon: %.2f sec\n\n", aTimer.GetElapsedSeconds());
	delete m_pTomoStack;
	m_pTomoStack = pVolStack;
}

void CProcessThread::mFlipInt(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iFlipInt == 0) return;
	MassNorm::CFlipInt3D aFlipInt;
	aFlipInt.DoIt(m_pTomoStack);
}

void CProcessThread::mSaveCentralSlices(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iVolZ <= 0) return;
	//------------------------------
	MrcUtil::CGenCentralSlices aGenCentralSlices;
	aGenCentralSlices.DoIt(m_pTomoStack);
	//-----------------------------------
	Util::CSaveTempMrc aSaveTempMrc;
	bool bGpu = true;
	char acMrcFile[256] = {'\0'};
	strcpy(acMrcFile, pInput->m_acOutMrcFile);
	int aiSize[2] = {0, 0};
	aGenCentralSlices.GetSizeXY(aiSize);
	aSaveTempMrc.SetFile(acMrcFile, "_projXY");
	aSaveTempMrc.DoIt(aGenCentralSlices.m_pfSliceXY, 2, aiSize);
	//---------------------------------------------------------
	//aGenCentralSlices.GetSizeYZ(aiSize);
	//aSaveTempMrc.OpenFile(acMrcFile, "_projYZ", aiSize, 1);
	//aSaveTempMrc.DoIt(0, aGenCentralSlices.m_pfSliceYZ, !bGpu);
	//-----------------------------------------------------------
	aGenCentralSlices.GetSizeXZ(aiSize);
	aSaveTempMrc.SetFile(acMrcFile, "_projXZ");
	aSaveTempMrc.DoIt(aGenCentralSlices.m_pfSliceXZ, 2, aiSize);
}

void CProcessThread::mFlipVol(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iFlipVol == 0) return;
	//---------------------------------
	printf("Flip volume from xzy view to xyz view.\n");
	int* piOldSize = m_pTomoStack->m_aiStkSize;
	int aiNewSize[] = {piOldSize[0], piOldSize[2], piOldSize[1]};
	MrcUtil::CTomoStack* pNewStack = new MrcUtil::CTomoStack;
	pNewStack->Create(aiNewSize, true);
	//---------------------------------
	int iBytes = aiNewSize[0] * sizeof(float);
	int iEndOldY = piOldSize[1] - 1;
	//------------------------------
	for(int y=0; y<piOldSize[1]; y++)
	{	float* pfDstFrm = pNewStack->GetFrame(iEndOldY - y);
		for(int z=0; z<piOldSize[2]; z++)
		{	float* pfSrcFrm = m_pTomoStack->GetFrame(z);
			memcpy(pfDstFrm + z * aiNewSize[0],
			  pfSrcFrm + y * aiNewSize[0], iBytes);
		}
		if((y % 100) != 0) continue;
		printf("...... %5d slices flipped, %5d left.\n",
			y + 1, piOldSize[1] - 1 - y);
	}
	delete m_pTomoStack;
	m_pTomoStack = pNewStack;
	printf("flip volume completed.\n\n");
}


void CProcessThread::mSaveAlignment(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iAlign == 0) return;
	//------------------------------- 
	MrcUtil::CLoadAlignment* pLoadAlign =
           MrcUtil::CLoadAlignment::GetInstance();
        if(pLoadAlign->m_bFromAlnFile) return;
	//------------------------------------	
	MrcUtil::CSaveAlignFile saveAlignFile;
	saveAlignFile.DoIt(pInput->m_acInMrcFile, pInput->m_acOutMrcFile, 
	   m_pAlignParam, m_pLocalParam);
}


void CProcessThread::mSaveStack(void)
{
	CInput* pInput = CInput::GetInstance();
	float* pfStats = new float[4];
	MrcUtil::CCalcStackStats::DoIt(m_pTomoStack, pfStats,
	   pInput->m_piGpuIDs, pInput->m_iNumGpus);
	//-----------------------------------------
	bool bVolume = true;
	if(pInput->m_iVolZ == 0) bVolume = false;
	float fPixelSize = pInput->m_fPixelSize * pInput->m_fOutBin;
	//----------------------------------------------------------
	MrcUtil::CSaveStack aSaveStack;
	aSaveStack.OpenFile(pInput->m_acOutMrcFile);
	aSaveStack.DoIt(m_pTomoStack, m_pAlignParam,
	   fPixelSize, pfStats, bVolume);
	//-------------------------------
	if(pfStats != 0L) delete[] pfStats;
}
