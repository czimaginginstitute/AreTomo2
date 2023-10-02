#include "CMainInc.h"
#include "../Util/CUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace TomoRalign;

CInput* CInput::m_pInstance = 0L;

CInput* CInput::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CInput;
	return m_pInstance;
}

void CInput::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CInput::CInput(void)
{
	strcpy(m_acInMrcTag, "-InMrc");
	strcpy(m_acOutMrcTag, "-OutMrc");
	strcpy(m_acVolSizeTag, "-VolSize");
	strcpy(m_acTmpFileTag, "-TmpFile");
	strcpy(m_acLogFileTag, "-LogFile");
	strcpy(m_acGpuIDTag, "-Gpu");
	//---------------------------
	memset(m_aiVolSize, 0, sizeof(m_aiVolSize));
	m_aiVolSize[2] = 256;
	m_piGpuIds = 0L;
	m_iNumGpus = 0;
}

CInput::~CInput(void)
{
	if(m_piGpuIds != 0L) delete[] m_piGpuIds;
}

void CInput::ShowTags(void)
{
	printf
	(  "%-10s\n"
	   "  1. Input MRC file that stores tomo tilt series.\n\n",
	   m_acInMrcTag
	);
	//-------------
        printf
	(  "%-10s\n"
	   "  1. Output MRC file that stores the aligned tilt series.\n\n",
	   m_acOutMrcTag
	);
	//--------------
        printf
	(  "%-10s\n"
	   "  1. Temporary image file for debugging.\n\n",
	   m_acTmpFileTag
	);
	//---------------
	printf
	(  "%-10s\n"
	   "  1. Log file storing alignment data.\n\n",
	   m_acLogFileTag
	);
	//---------------
        printf("%-10s\n", m_acVolSizeTag);
	printf("   x, y,  z dimension of volume to be reconstructed, \n");
	printf("   By default, full projection will be reconstructed \n");
	printf("   to a box of 256 pixel in z dimension.\n\n");
	//-----------------------------------------------------
        printf("%-10s\n", m_acGpuIDTag);
	printf("   GPU IDs. Default 0. Currently only 1 GPU is\n");
	printf("   supported.\n\n");
}

void CInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//------------
	memset(m_acInMrcFile, 0, sizeof(m_acInMrcFile));
	memset(m_acOutMrcFile, 0, sizeof(m_acOutMrcFile));
	memset(m_acTmpFile, 0, sizeof(m_acTmpFile));
	memset(m_acLogFile, 0, sizeof(m_acLogFile));
	//------------------------------------------
	int aiRange[2];
	Util::CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	aParseArgs.FindVals(m_acInMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInMrcFile);
	//-------------------------------------------
	aParseArgs.FindVals(m_acOutMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acOutMrcFile);
	//--------------------------------------------
	aParseArgs.FindVals(m_acTmpFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acTmpFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acLogFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acLogFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acVolSizeTag, aiRange);
	aParseArgs.GetVals(aiRange, m_aiVolSize);
	//---------------------------------------
	if(m_piGpuIds != 0L) delete[] m_piGpuIds;
	aParseArgs.FindVals(m_acGpuIDTag, aiRange);
	if(aiRange[1] >= 1)
	{	m_iNumGpus = aiRange[1];
		m_piGpuIds = new int[m_iNumGpus];
		aParseArgs.GetVals(aiRange, m_piGpuIds);
	}
	else
	{	m_iNumGpus = 1;
		m_piGpuIds = new int[m_iNumGpus];
		m_piGpuIds[0] = 0;
	}
	//------------------------
	mPrint();	
}

void CInput::mPrint(void)
{
	printf("\n");
	printf("%-10s  %s\n", m_acInMrcTag, m_acInMrcFile);
	printf("%-10s  %s\n", m_acOutMrcTag, m_acOutMrcFile);
	printf("%-10s  %s\n", m_acTmpFileTag, m_acTmpFile);
	printf("%-10s  %s\n", m_acLogFileTag, m_acLogFile);
	printf
	(  "%-10s  %d  %d  %d\n", m_acVolSizeTag, 
	   m_aiVolSize[0], m_aiVolSize[1], m_aiVolSize[2]
	);
	printf("%-10s", m_acGpuIDTag);
	for(int i=0; i<m_iNumGpus; i++) 
	{	printf("  %d", m_piGpuIds[i]);
	}
	printf("\n\n");
}
