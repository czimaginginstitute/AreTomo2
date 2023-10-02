#include "CCorrectInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace Correct;

CBinStack::CBinStack(void)
{
}

CBinStack::~CBinStack(void)
{
}

void CBinStack::DoIt
(	MrcUtil::CTomoStack* pTomoStack,
	int iBinning,
	MrcUtil::CTomoStack* pBinStack	
)
{	bool bInPadded = true, bOutPadded = true;
	int* piInSize = pTomoStack->m_aiStkSize;
	int aiOutSize[] = {0, 0, piInSize[2]};
	aiOutSize[2] = pTomoStack->m_aiStkSize[2];
	Util::GBinImage2D::GetBinSize(piInSize, !bInPadded, 
		iBinning, aiOutSize, !bOutPadded);
	pBinStack->Create(aiOutSize, true);
	//---------------------------------
	Util::GBinImage2D aGBinImg2D;
	aGBinImg2D.SetupBinning(piInSize, iBinning, !bInPadded, !bOutPadded); 
	//-------------------------------------------------------------------
	float* gfOutImg = 0L;
	size_t tOutBytes = sizeof(float) * pBinStack->GetPixels();
	cudaMalloc(&gfOutImg, tOutBytes);
	//-------------------------------
	float* gfInImg = 0L;
	size_t tInBytes = sizeof(float) * pTomoStack->GetPixels();
	cudaMalloc(&gfInImg, tInBytes);
	//-----------------------------
	for(int i=0; i<pBinStack->m_aiStkSize[2]; i++)
	{	float* pfProj = pTomoStack->GetFrame(i);
		cudaMemcpy(gfInImg, pfProj, tInBytes, cudaMemcpyDefault);
		//-------------------------------------------------------
		aGBinImg2D.DoIt(gfInImg, gfOutImg);
		float* pfBinProj = pBinStack->GetFrame(i);
		cudaMemcpy(pfBinProj, gfOutImg, tOutBytes, cudaMemcpyDefault);
	}
	if(gfInImg != 0L) cudaFree(gfInImg);
	if(gfOutImg != 0L) cudaFree(gfOutImg);
}
/*
void CBinStack::DoProj
(	float* pfProj,
	int* piProjSize,
	int iBinning,
	float* pfBinProj
)
{	bool bPadded = true;
	Util::GBinImage2D aGBinImg2D;
	int aiBinning[] = {iBinning, iBinning};
	aGBinImg2D.Setup(piProjSize, !bPadded, aiBinning);
	//------------------------------------------------
	bool bZero = true;
	int aiBinSize[2] = {0, 0};
	aiBinSize[0] = aGBinImg2D.m_aiBinSize[0];
	aiBinSize[1] = aGBinImg2D.m_aiBinSize[1];
	float* gfBinProj = aGBinImg2D.GetGBuf(aiBinSize, !bZero);
	//-------------------------------------------------------
	aGBinImg2D.DoIt(pfProj, gfBinProj, !bPadded);
	size_t tBytes = sizeof(float) * aiBinSize[0] * aiBinSize[1];
	cudaMemcpy(pfBinProj, gfBinProj, tBytes, cudaMemcpyDefault);
	cudaFree(gfBinProj);
}
*/
