#include "CImodUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../FindCtf/CFindCtfInc.h"
#include "../CInput.h"
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <limits.h>

using namespace ImodUtil;

CImodUtil* CImodUtil::m_pInstance = 0L;

CImodUtil* CImodUtil::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CImodUtil;
	return m_pInstance;
}

void CImodUtil::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CImodUtil::CImodUtil(void)
{
}

CImodUtil::~CImodUtil(void)
{
}

void CImodUtil::CreateFolder(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iOutImod == 0) return;
	//---------------------------------
	Util::CFileName aInFileName(pInput->m_acInMrcFile);
	Util::CFileName aOutFileName(pInput->m_acOutMrcFile);
	char acPrefix[256] = {'\0'};
	aInFileName.GetName(acPrefix);
	aOutFileName.GetFolder(m_acOutFolder);
	//-------------------------------------------------------------
	// Create a subfolder in the output folder to store Imod files.
	// Its name is the output MRC file name appended with _Imod.
	//-------------------------------------------------------------
	strcat(m_acOutFolder, acPrefix);
	strcat(m_acOutFolder, "_Imod/");
	struct stat st = {0};
        if (stat(m_acOutFolder, &st) == -1)
        {       mkdir(m_acOutFolder, 0700);
        }	
	//---------------------------------
	strcpy(m_acTltFile, acPrefix);
	strcat(m_acTltFile, "_st.tlt");
	//--------------------------
	strcpy(m_acCsvFile, acPrefix);
	if(pInput->m_iOutImod == 1) strcat(m_acCsvFile, "_order_list.csv");
	else strcat(m_acCsvFile, "_st_order_list.csv");
	//---------------------------------------------
	strcpy(m_acXfFile, acPrefix);
	strcat(m_acXfFile, "_st.xf");
	//------------------------
	strcpy(m_acAliFile, acPrefix);
	strcat(m_acAliFile, "_st.ali");
	//--------------------------
	strcpy(m_acXtiltFile, acPrefix);
	strcat(m_acXtiltFile, "_st.xtilt");
	//------------------------------
	strcpy(m_acRecFile, acPrefix);
	strcat(m_acRecFile, "_st.rec");
	//-----------------------------
	strcpy(m_acCtfFile, acPrefix);
	strcat(m_acCtfFile, "_ctf.txt");
	//------------------------------
	if(pInput->m_iOutImod == 1) 
	{	char acBuf[25] = {'\0'};	
		char* pcRes = realpath(pInput->m_acInMrcFile, acBuf);
		if(pcRes != 0L) strcpy(m_acInMrcFile, acBuf);
		else strcpy(m_acInMrcFile, pInput->m_acInMrcFile);
	}
	else if(pInput->m_iOutImod >= 2)
	{	strcpy(m_acInMrcFile, acPrefix);
		strcat(m_acInMrcFile, "_st.mrc");
	}
}

void CImodUtil::SaveTiltSeries
(	MrcUtil::CTomoStack* pTomoStack,
	MrcUtil::CAlignParam* pAlignParam,
	float fPixelSize
)
{	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iOutImod == 0) return;
	//---------------------------------
	m_pTomoStack = pTomoStack; // can be NULL
	m_pGlobalParam = pAlignParam;
	m_fPixelSize = fPixelSize;
	m_fTiltAxis = m_pGlobalParam->GetTiltAxis(
           m_pGlobalParam->m_iNumFrames / 2);
	//----------------------------------
	char acFile[256] = {'\0'};
	mCreateFileName(m_acTltFile, acFile);
	CSaveTilts aSaveTilts;
	aSaveTilts.DoIt(pAlignParam, acFile);
	//-----------------------------------
	mCreateFileName(m_acCsvFile, acFile);
	MrcUtil::CAcqSequence* pAcqSequence =
	   MrcUtil::CAcqSequence::GetInstance();
	pAcqSequence->SaveCsv(acFile, pInput->m_iOutImod);
	//------------------------------------------------
	mCreateFileName(m_acXfFile, acFile);
	CSaveXF aSaveXF;
	aSaveXF.DoIt(pAlignParam, acFile);
	//--------------------------------	
	mCreateFileName(m_acXtiltFile, acFile);
	CSaveXtilts aSaveXtilts;
	aSaveXtilts.DoIt(pAlignParam, acFile);
	//------------------------------------
	mSaveTiltSeries();
	mSaveNewstComFile();
	mSaveTiltComFile();
	mSaveCtfFile();
}

void CImodUtil::mSaveTiltSeries(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iOutImod != 2 && pInput->m_iOutImod != 3) return;
	if(m_pTomoStack == 0L) return;
	//----------------------------
	char acFile[256];
	mCreateFileName(m_acInMrcFile, acFile);
	//----------------------------------
	MrcUtil::CSaveStack aSaveStack;
	bool bOpen = aSaveStack.OpenFile(acFile);
	if(!bOpen) return;
	//----------------
	printf("Save tilt series, please wait .....\n");
	bool bVolume = true;
	aSaveStack.DoIt(m_pTomoStack, m_pGlobalParam, 
	   m_fPixelSize, 0L, !bVolume);
	printf("Aligned series has been saved.\n");
}

void CImodUtil::mSaveNewstComFile(void)
{
	char acComFile[256];
	mCreateFileName("newst.com", acComFile);
	FILE* pFile = fopen(acComFile, "wt");
	if(pFile == 0L) return;
	//---------------------
	fprintf(pFile, "$newstack -StandardInput\n");
	fprintf(pFile, "InputFile	%s\n", m_acInMrcFile);
	fprintf(pFile, "OutputFile	%s\n", m_acAliFile);
	fprintf(pFile, "TransformFile	%s\n", m_acXfFile);     
	fprintf(pFile, "TaperAtFill     1,0\n");
	fprintf(pFile, "AdjustOrigin\n");
	fprintf(pFile, "OffsetsInXandY  0.0,0.0\n");
	fprintf(pFile, "#DistortionField        .idf\n");
	fprintf(pFile, "ImagesAreBinned 1.0\n");
	fprintf(pFile, "BinByFactor     1\n");
	fprintf(pFile, "#GradientFile   hc20211206_804.maggrad\n");
	fprintf(pFile, "$if (-e ./savework) ./savework");
	//-----------------------------------------------
	fclose(pFile);	
}

void CImodUtil::mSaveTiltComFile(void)
{
	char acComFile[256];
	mCreateFileName("tilt.com", acComFile);
	FILE* pFile = fopen(acComFile, "wt");
	if(pFile == 0L) return;
	//---------------------
	MrcUtil::CDarkFrames* pDarkFrames = 0L;
	pDarkFrames = MrcUtil::CDarkFrames::GetInstance();
	int* piRawSize = pDarkFrames->m_aiRawStkSize;
	int aiAlnSize[] = {0, 0};
	Correct::CCorrectUtil::CalcAlignedSize(piRawSize, 
	   m_fTiltAxis, aiAlnSize);
	CInput* pInput = CInput::GetInstance();
	//-----------------------------------------------
	fprintf(pFile, "$tilt -StandardInput\n");
	fprintf(pFile, "InputProjections %s\n", m_acAliFile);
	fprintf(pFile, "OutputFile %s\n", m_acRecFile);
	fprintf(pFile, "IMAGEBINNED 1\n");
	fprintf(pFile, "TILTFILE %s\n", m_acTltFile);
	fprintf(pFile, "THICKNESS %d\n", pInput->m_iVolZ);
	fprintf(pFile, "RADIAL 0.35 0.035\n");
	fprintf(pFile, "FalloffIsTrueSigma 1\n");
	fprintf(pFile, "XAXISTILT 0.0\n");
	fprintf(pFile, "LOG 0.0\n");
	fprintf(pFile, "SCALE 0.0 250.0\n");
	fprintf(pFile, "PERPENDICULAR\n");
	fprintf(pFile, "Mode 2\n");
	fprintf(pFile, "FULLIMAGE %d %d\n", aiAlnSize[0], aiAlnSize[1]);
	fprintf(pFile, "SUBSETSTART 0 0\n");
	fprintf(pFile, "AdjustOrigin\n");
	fprintf(pFile, "LOCALFILE %s\n", m_acXfFile);
	fprintf(pFile, "ActionIfGPUFails 1,2\n");
	fprintf(pFile, "XTILTFILE %s\n", m_acXtiltFile);
	fprintf(pFile, "OFFSET 0.0\n");
	fprintf(pFile, "SHIFT 0.0 0.0\n");
	//--------------------------------
	if(pInput->m_iOutImod == 1 && pDarkFrames->m_iNumDarks > 0)
	{	char acExclude[128] = {'\0'};
		pDarkFrames->GenImodExcludeList(acExclude, 128);
		fprintf(pFile, "%s\n", acExclude);
	}
	fprintf(pFile, "$if (-e ./savework) ./savework");
	//-----------------------------------------------
	fclose(pFile);	
}

void CImodUtil::mSaveCtfFile(void)
{
	CInput* pInput = CInput::GetInstance();
	FindCtf::CCtfResults* pCtfResults = 
	   FindCtf::CCtfResults::GetInstance();
	if(pCtfResults->m_iNumImgs <= 0) return;
	//--------------------------------------
	char acFile[256] = {'\0'};	
	mCreateFileName(m_acCtfFile, acFile);
	FILE* pFile = fopen(acFile, "w");
	if(pFile == 0L) return;
 	//--------------------------------------
	float fExtPhase = pCtfResults->GetExtPhase(0);
	if(fExtPhase == 0) fprintf(pFile, "1  0  0.0  0.0  0.0  3\n");
	else fprintf(pFile, "5  0  0.0  0.0  0.0  3\n");
	//----------------------------------------------
	const char *pcFormat1 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f\n";
	const char *pcFormat2 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f  %8.2f\n";
	float fDfMin, fDfMax;
	if(fExtPhase == 0)
	{	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
		{	float fTilt = pCtfResults->GetTilt(i);
			fDfMin = pCtfResults->GetDfMin(i) * 0.1f;
			fDfMax = pCtfResults->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat1, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, pCtfResults->GetAzimuth(i));
		}
	}
	else
	{	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
		{	float fTilt = pCtfResults->GetTilt(i);
			fDfMin = pCtfResults->GetDfMin(i) * 0.1f;
			fDfMax = pCtfResults->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat2, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, pCtfResults->GetAzimuth(i),
			   pCtfResults->GetExtPhase(i));
		}
	}
	fclose(pFile);
}

void CImodUtil::mCreateFileName(const char* pcInFileName, char* pcOutFileName)
{
	strcpy(pcOutFileName, m_acOutFolder);
	strcat(pcOutFileName, pcInFileName);
}
