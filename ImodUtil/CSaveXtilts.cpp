#include "CImodUtilInc.h"
#include "../CInput.h"
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace ImodUtil;

CSaveXtilts::CSaveXtilts(void)
{
}

CSaveXtilts::~CSaveXtilts(void)
{
}

void CSaveXtilts::DoIt
(	MrcUtil::CAlignParam* pAlignParam,
	const char* pcFileName
)
{	FILE* pFile = fopen(pcFileName, "wt");
	if(pFile == 0L) return;
	//-----------------
	CInput* pInput = CInput::GetInstance();
	MrcUtil::CDarkFrames* pDarkFrames = 0L;
	pDarkFrames = MrcUtil::CDarkFrames::GetInstance();
	//-----------------
	int iTilts = pAlignParam->m_iNumFrames;
	if(pInput->m_iOutImod == 1) 
	{	iTilts = pDarkFrames->m_aiRawStkSize[2];
	}
	//-----------------
	int iLast = iTilts - 1;
	for(int i=0; i<iLast; i++)
	{	fprintf(pFile, "0\n");
	}
	fclose(pFile);
}
