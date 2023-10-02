#include "CFindCtfInc.h"
#include "../CInput.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace FindCtf;

CFindCtfMain::CFindCtfMain(void)
{
	m_ppfHalfSpects = 0L;
	m_pFindCtf2D = 0L;
	m_iNumTilts = 0;
	m_iRefTilt = 0;
}

CFindCtfMain::~CFindCtfMain(void)
{
	this->Clean();
}

void CFindCtfMain::Clean(void)
{
	if(m_ppfHalfSpects != 0L)
	{	for(int i=0; i<m_iNumTilts; i++)
		{	if(m_ppfHalfSpects[i] == 0L) continue;
			else delete[] m_ppfHalfSpects[i];
		}
		delete[] m_ppfHalfSpects;
		m_ppfHalfSpects = 0L;
	}
	if(m_pFindCtf2D != 0L)
	{	delete m_pFindCtf2D;
		m_pFindCtf2D = 0L;
	}
	m_iNumTilts = 0;
}

bool CFindCtfMain::CheckInput(void)
{
	CInput* pInput = CInput::GetInstance();
	bool bEstimate = true;
	if(pInput->m_fCs == 0.0) bEstimate = false;
	else if(pInput->m_fKv == 0) bEstimate = false;
	else if(pInput->m_fPixelSize == 0) bEstimate = false;
	if(bEstimate) return true;
	//------------------------
	printf("Skip CTF estimation. Need the following parameters.\n");
	printf("High tension: %f\n", pInput->m_fKv);
	printf("Cs value:     %f\n", pInput->m_fCs);
	printf("Pixel size:   %f\n\n", pInput->m_fPixelSize);
	return false;	
}

void CFindCtfMain::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam
)
{	this->Clean();
	//------------
	m_pTomoStack = pTomoStack;
	m_pAlignParam = pAlignParam;
	m_iNumTilts = m_pTomoStack->m_aiStkSize[2];
	//-----------------------------------------
	CInput* pInput = CInput::GetInstance();
	CCtfResults* pCtfResults = CCtfResults::GetInstance();
	int aiSpectSize[] = {512, 512};
	pCtfResults->Setup(m_pTomoStack->m_aiStkSize[2], aiSpectSize);
	//-------------------------------------------------------------
	float fExtPhase = pInput->m_afExtPhase[0] * 0.017453f;
	CCtfTheory aInputCTF;
	aInputCTF.Setup(pInput->m_fKv, pInput->m_fCs,
	   pInput->m_fAmpContrast, pInput->m_fPixelSize,
	   100.0f, fExtPhase);
	//----------------------------------------------
	m_pFindCtf2D = new CFindCtf2D;
	m_pFindCtf2D->Setup1(&aInputCTF);
	m_pFindCtf2D->Setup2(m_pTomoStack->m_aiStkSize);
	m_pFindCtf2D->SetPhase(pInput->m_afExtPhase[0], 
	   pInput->m_afExtPhase[1]);
	//--------------------------
	mGenSpectrums();
	mDoZeroTilt();
	mDo2D();
	//--------------
	CSaveCtfResults* pSaveCtfResults = CSaveCtfResults::GetInstance();
	pSaveCtfResults->AsyncSave();
	//mSaveSpectFile();
}

void CFindCtfMain::mGenSpectrums(void)
{
	bool bRaw = true, bToHost = true;
	if(m_ppfHalfSpects == 0L) m_ppfHalfSpects = new float*[m_iNumTilts];
	//------------------------------------------------------------------
	for(int i=0; i<m_iNumTilts; i++)
	{	printf("......   spectrum of image %4d created, %4d left\n",
		   i+1, m_iNumTilts - 1 - i);	
		float* pfImage = m_pTomoStack->GetFrame(i);
		m_pFindCtf2D->GenHalfSpectrum(pfImage);
		m_ppfHalfSpects[i] = m_pFindCtf2D->GetHalfSpect(!bRaw, bToHost);
	}
	printf("\n");
}

void CFindCtfMain::mDoZeroTilt(void)
{
	m_iRefTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	//----------------------------------------------------
	CInput* pInput = CInput::GetInstance();
        float fPhaseRange = fmaxf(pInput->m_afExtPhase[1], 0.0f);
	m_pFindCtf2D->SetPhase(pInput->m_afExtPhase[0], fPhaseRange);
	//-----------------------------------------------------------
	m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[m_iRefTilt]);
	m_pFindCtf2D->Do2D();
	//-------------------
	mGetResults(m_iRefTilt);
}
	   
void CFindCtfMain::mDo2D(void)
{	
	float afDfRange[2] = {0.0f}, afAstRatio[2] = {0.0f};
	float afAstAngle[2] = {0.0f}, afExtPhase[2] = {0.0f};
	//---------------------------------------------------
	CCtfResults* pCtfResults = CCtfResults::GetInstance();
	float fDfMin = pCtfResults->GetDfMin(m_iRefTilt);
	float fDfMax = pCtfResults->GetDfMax(m_iRefTilt);
	afDfRange[0] = 0.5f * (fDfMin + fDfMax); 
	afDfRange[1] = fmaxf(afDfRange[0] * 0.25f, 5000.0f);
	afAstRatio[0] = CFindCtfHelp::CalcAstRatio(fDfMin, fDfMax);
	//---------------------------------------------------------
	afAstAngle[0] = pCtfResults->GetAzimuth(m_iRefTilt);
	afExtPhase[0] = pCtfResults->GetExtPhase(m_iRefTilt);
	//---------------------------------------------------
	for(int i=0; i<m_pTomoStack->m_aiStkSize[2]; i++)
	{	if(i == m_iRefTilt) continue;
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		m_pFindCtf2D->Refine(afDfRange, afAstRatio, 
		   afAstAngle, afExtPhase);
		mGetResults(i);
	}
	//---------------------
	CInput* pInput = CInput::GetInstance();
	printf("Estimated CTFs:  %s\n", pInput->m_acInMrcFile);
	printf("Index  dfmin     dfmax    azimuth  phase   score\n");
	pCtfResults->DisplayAll();
	printf("\n");
}

void CFindCtfMain::mSaveSpectFile(void)
{
	char* pcSpectFileName = mGenSpectFileName();
	//------------------------------------------
	Mrc::CSaveMrc aSaveMrc;
	if(!aSaveMrc.OpenFile(pcSpectFileName))
	{	printf("Info: unable to open %s\n", pcSpectFileName);
		printf("Info: spectrums in CTF estimation not saved.\n\n");
		if(pcSpectFileName != 0L) delete[] pcSpectFileName;
		return;
	}
	//---------------------------------------------------------
	int aiSpectSize[2] = {0};
	m_pFindCtf2D->GetSpectSize(aiSpectSize, false);
	aSaveMrc.SetMode(Mrc::eMrcFloat);
	aSaveMrc.SetImgSize(aiSpectSize, m_iNumTilts, 1, 1.0f);
	aSaveMrc.SetExtHeader(0, 32, 0);
	aSaveMrc.m_pSaveMain->DoIt();
	//---------------------------
	CCtfResults* pCtfResults = CCtfResults::GetInstance();
	//----------------------------------------------------
	for(int i=0; i<m_iNumTilts; i++)
	{	printf("......  saving spectrum %4d, %4d left\n",
		   i+1, m_iNumTilts - 1 - i);
		float fTilt = m_pAlignParam->GetTilt(i);
		aSaveMrc.m_pSaveExt->SetTilt(i, &fTilt, 1);
		aSaveMrc.m_pSaveExt->DoIt();
		//-----------------------------------------
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		//-------------------------------------------
		m_pFindCtf2D->m_fDfMin = pCtfResults->GetDfMin(i);
		m_pFindCtf2D->m_fDfMax = pCtfResults->GetDfMax(i);
		m_pFindCtf2D->m_fAstAng = pCtfResults->GetAzimuth(i);
		m_pFindCtf2D->m_fExtPhase = pCtfResults->GetExtPhase(i);
	}
	printf("\n");
}

void CFindCtfMain::mGetResults(int iTilt)
{
	CCtfResults* pCtfResults = CCtfResults::GetInstance();
	float fTilt = m_pAlignParam->GetTilt(iTilt);
	//------------------------------------------
	pCtfResults->SetTilt(iTilt, fTilt);
	pCtfResults->SetDfMin(iTilt, m_pFindCtf2D->m_fDfMin);
	pCtfResults->SetDfMax(iTilt, m_pFindCtf2D->m_fDfMax);
	pCtfResults->SetAzimuth(iTilt, m_pFindCtf2D->m_fAstAng);
	pCtfResults->SetExtPhase(iTilt, m_pFindCtf2D->m_fExtPhase);
	pCtfResults->SetScore(iTilt, m_pFindCtf2D->m_fScore);
	//---------------------------------------------------
	float* pfSpect = m_pFindCtf2D->GenFullSpectrum();
	pCtfResults->SetSpect(iTilt, pfSpect);
}

char* CFindCtfMain::mGenSpectFileName(void)
{
	CInput* pInput = CInput::GetInstance();
	Util::CFileName aInMrcFile, aOutMrcFile;
	aInMrcFile.Setup(pInput->m_acInMrcFile);
	aOutMrcFile.Setup(pInput->m_acOutMrcFile);
	//----------------------------------------
	char acBuf[256] = {'\0'};
	char* pcSpectName = new char[256];
	memset(pcSpectName, 0, sizeof(char) * 256);
	//-----------------------------------------
	aInMrcFile.GetName(acBuf);
	aOutMrcFile.GetFolder(pcSpectName);
	strcat(pcSpectName, acBuf);
	strcat(pcSpectName, "_CTF.mrc");
	return pcSpectName;
}	

