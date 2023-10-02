#include "GGenWeight.h"
#include <memory.h>
#include <stdio.h>

__device__ __constant__ int giVolSize[2];	// iVolX, iVolZ
__device__ __constant__ int giBoxSize[2];	// iBoxX, iBoxZ
//---------------------------------------

static __global__ void mGGenWeights
(	int iProjX,
	float* gfTilts
	float* gfRawProjWeight,
	float* gfForProjWeight
)
{	int iXp = blockIdx.x * blockDim.x + threadIdx.x;
	if(iXp >= iProjX) return;	
	//-----------------------
	float fCos = gfTilts[blockIdx.y] * 3.141593f / 180.0f;
	float fSin = sinf(fCos);
	fCos = cosf(fCos);
	//----------------
	int iRayLength = (int)(giVolSize[0] * fabsf(fSin) 
	+ giVolSize[1] * fCos) / 2 * 2;;
	//------------------------------
	float fTempX = (iXp - (iProjX - 1) * 0.5f) * fCos 
		+ (giVolSize[0] - 1) * 0.5f; 
	float fTempZ = (iXp - (iProjX - 1) * 0.5f) * fSin
		+ (giVolSize[1] - 1) * 0.5f;
	//----------------------------------
	float fWeight = 0.0f;
	for(int i=0; i<iRayLength; i++)
	{	float fZ = i - (iRayLenght - 1) * 0.5f;
		float fX = fTempX - fZ * fSin;
		fZ = fTempZ + fZ * fCosSin.x;
		if(fX < 0 || fX > iEndX) continue;
		else if(fZ < 0 || fZ > iEndZ) continue;
		fWeight += 1.0f;
	}
	int i = blockIdx.y * giProjSize[0] + iXp;
	gfWeight[i] = fWeight;
}

//-------------------------------------------------------------------
// 1. aPitchedPtr has the dimension of [iBoxSizeX, iNumProjs].
// 2. blockIdx.y stores the projection index iProj that is used
//    to retrieve cosine and sine of tilt angle.
//-------------------------------------------------------------------
static __global__ void mPreWeight
(	int iBoxSizeX,
	int iBoxSizeZ,
	float fDefaultWeight,
	cudaPitchedPtr aPitchedPtr
)
{	int iXr = blockIdx.x * blockDim.x + threadIdx.x;
	if(iXr >= iBoxSizeX) return;
	// -------------------------
	float2 fCosSin = tex2D(texCosSin, blockIdx.y, 0);
	int iSizeZ = (int)(iBoxSizeX * fabsf(fCosSin.y) 
		+ iBoxSizeZ * fCosSin.x);
	iSizeZ = (iSizeZ / 2) * 2 + 2;
	//----------------------------
	float fCentX = iBoxSizeX * 0.5f;
	float fCentZ = iSizeZ * 0.5f;
	//---------------------------
	float fTempX = (iXr - fCentX) * fCosSin.x + fCentX; 
	float fTempZ = (iXr - fCentX) * fCosSin.y + fCentZ;
	//-------------------------------------------------
	int iWeight = 0;
	for(int z=0; z<iSizeZ; z++)
	{	float fX = fTempX - (z - fCentZ) * fCosSin.y;
		float fZ = fTempZ + (z - fCentZ) * fCosSin.x;
		if(fX < 0 || fZ < 0) continue;
		if(fX > (iBoxSizeX - 1)) continue;
		if(fZ > (iBoxSizeZ - 1)) continue;
		iWeight++;
	}
	//----------------
	float* pfVal = (float*)((char*)aPitchedPtr.ptr 
		+ blockIdx.y * aPitchedPtr.pitch);
	if(iWeight > 0) pfVal[iXr] = 1.0f / iWeight;
	else pfVal[iXr] = fDefaultWeight;
}

GGenWeight::GGenWeight(void)
{
	memset(m_aiVolSize, 0, sizeof(m_aiVolSize));
	memset(m_aiProjSize, 0, sizeof(m_aiProjSize));
	memset(m_aiBoxSize, 0, sizeof(m_aiBoxSize));
	texCosSin.filterMode = cudaFilterModePoint;
}

GGenWeight::~GGenWeight(void)
{
	cudaUnbindTexture(texCosSin);
}

void GGenWeight::SetVolSize(int iVolX, int iVolZ)
{
	if(m_aiVolSize[0] == iVolX && m_aiVolSize[1] == iVolZ) return;
	m_aiVolSize[0] = iVolX;
	m_aiVolSize[1] = iVolZ;
	cudaMemcpyToSymbol(giVolSize, m_aiVolSize, sizeof(int) * 2);
}

void GGenWeight::SetProjSize(int iProjX, int iProjs)
{
	if(m_aiProjSize[0] == iProjX && m_aiProjSize[1] == iProjs) return;
	m_aiProjSize[0] = iProjX;
	m_aiProjSize[1] = iProjs;
	cudaMemcpyToSymbol(giProjSize, m_aiProjSize, sizeof(int) * 2);
}

void GGenWeight::SetBoxSize(int iBoxX, int iBoxZ)
{
	if(m_aiBoxSize[0] == iBoxX && m_aiBoxSize[1] == iBoxZ) return;
	m_aiBoxSize[0] = iBoxX;
	m_aiBoxSize[1] = iBoxZ;
	cudaMemcpyToSymbol(giBoxSize, m_aiBoxSize, sizeof(int) * 2);
}

void GGenWeight::SetCosSinArray(cudaArray* pArray)
{
	cudaBindTextureToArray(texCosSin, pArray);
}

void GGenWeight::DoProjWeight(cudaPitchedPtr aPitchedPtr)
{	
	dim3 aBlockDim(256, 1);
	dim3 aGridDim(1, m_aiProjSize[1]);
	if((aGridDim.x * aBlockDim.x) < m_aiProjSize[0]) aGridDim.x += 1;	
	mProjWeight<<<aGridDim, aBlockDim>>>(aPitchedPtr);
	cudaThreadSynchronize();
}

void GGenWeight::DoPreWeight
(	int* piBoxSize,
	int iNumProjs,
	float fDefaultWeight,
	cudaPitchedPtr aPitchedPtr
)
{	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, iNumProjs);
	aGridDim.x = piBoxSize[0] / aBlockDim.x + 1;
	mPreWeight<<<aGridDim, aBlockDim>>>
	(  piBoxSize[0], piBoxSize[1],
	   fDefaultWeight,
	   aPitchedPtr
	);
}

