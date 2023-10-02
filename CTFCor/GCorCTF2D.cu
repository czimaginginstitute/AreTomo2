#include "CCTFCorInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace CTFCor;

extern __shared__ char s_acArray[];

static __global__ void mGSumPower
(	int iCmpX,
	int iCmpY,
	float fStartR,
	float fEndR,
	cufftComplex* gCmpImg,
	float* gfSum,
	float* gfCount
)
{	float* sfSum = (float*)&s_acArray[0];
	float* sfCount = (float*)&sfSum[blockDim.x];
	sfSum[threadIdx.x] = 0.0f;
	sfCount[threadIdx.x] = 0;
	__syncthreads();
	//--------------
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int y = i / iCmpX;
	if(y >= iCmpY) return;
	//--------------------
	float fX = (i % iCmpX) * 0.5f / (iCmpX - 1);
	float fY = ((float)y) / iCmpY;
	if(fY > 0.5f) fY -= 1.0f;
	//-----------------------
	float fR = sqrtf(fX * fX + fY * fY);
	if(i > 0 && fR > fStartR && fR < fEndR)
	{	fX = gCmpImg[i].x;
		fY = gCmpImg[i].y;
		sfSum[threadIdx.x] = sqrtf(fX * fX + fY * fY); 
		sfCount[threadIdx.x] += 1.0f;
	}
	__syncthreads();
	//--------------
	i = blockDim.x >> 1;
	while(i > 0)
	{	if(threadIdx.x < i)
		{	sfSum[threadIdx.x] += sfSum[threadIdx.x + i];
			sfCount[threadIdx.x] += sfCount[threadIdx.x + i];
		}
		__syncthreads();
		i = i >> 1;
	}
	//-----------------
	if(threadIdx.x != 0) return;
	gfSum[blockIdx.x] = sfSum[0];
	gfCount[blockIdx.x] = sfCount[0];
}

static __global__ void mGSum1D
(	int iSize,
	float* gfData,
	float* gfRes
)
{	float* sfRes = (float*)&s_acArray[0];
	sfRes[threadIdx.x] = 0.0f;
	__syncthreads();
	//--------------
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	//--------------------
	sfRes[threadIdx.x] = gfData[i];
	__syncthreads();
	//--------------
	i = blockDim.x >> 1;
	while(i > 0)
	{	if(threadIdx.x < i)
		{	sfRes[threadIdx.x] += sfRes[threadIdx.x + i];
		}
		__syncthreads();
		i = i >> 1;
	}
	//-----------------
	if(threadIdx.x != 0) return;
	gfRes[blockIdx.x] = sfRes[0];
}

static __global__ void mGCorCTF2D
(	int iCmpX,
	int iCmpY,
	float fSNR,
	float* gfCTF,
	float fFreq0,
	cufftComplex* gCmpImg
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float fX = i % iCmpX;
	float fY = i / iCmpX;
	if(i == 0 || fY >= iCmpY) return;
	//-------------------------------
	fX = fX / (iCmpX - 1) * 0.5f;
	fY = fY / iCmpY;
	if(fY > 0.5f) fY -= 1.0f;
	fX = sqrtf(fX * fX + fY * fY);
	//----------------------------
	float fFilt = gfCTF[i] / (gfCTF[i] * gfCTF[i] + 1.0f / fSNR);
	if(fX <= fFreq0) fFilt = 1.0f;
	gCmpImg[i].x *= fFilt;
	gCmpImg[i].y *= fFilt;
}

GCorCTF2D::GCorCTF2D(void)
{
	m_gfSumPower = 0L;
	m_gfCount = 0L;
	m_fPI = (float)(4.0 * atan(1.0));
	m_aBlockDim.x = 512;
}

GCorCTF2D::~GCorCTF2D(void)
{
	this->Clean();
}

void GCorCTF2D::Clean(void)
{
	if(m_gfSumPower != 0L) cudaFree(m_gfSumPower);
	if(m_gfCount != 0L) cudaFree(m_gfCount);
	m_gfSumPower = 0L;
	m_gfCount = 0L;
}

void GCorCTF2D::SetSize(int iCmpX, int iCmpY)
{
	this->Clean();
	//------------
	m_aiCmpSize[0] = iCmpX;
	m_aiCmpSize[1] = iCmpY;
	//---------------------
	int iSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	m_aGridDim.x = iSize / m_aBlockDim.x + 1;
	//---------------------------------------
	size_t tBytes = sizeof(float) * m_aGridDim.x;
	cudaMalloc(&m_gfSumPower, tBytes);
	cudaMalloc(&m_gfCount, tBytes);
}

void GCorCTF2D::DoIt
(	cufftComplex* gCmpImg,
	float* gfCTF,
	float fFreq0
)
{	m_gCmpImg = gCmpImg;
	//------------------
	float fSignal = mCalcPower(0.0f, 0.3f);
	float fNoise = mCalcPower(0.4, 1.0f);
	float fSNR = fSignal / fNoise;
	//----------------------------
	mGCorCTF2D<<<m_aGridDim, m_aBlockDim>>>
	( m_aiCmpSize[0], m_aiCmpSize[1],
	  fSNR, gfCTF, fFreq0, m_gCmpImg
	);
	//------------------------------
	m_gCmpImg = 0L;	   		
}

float GCorCTF2D::mCalcPower(float fStartFreq, float fEndFreq)
{	
	int iBytes = sizeof(float) * m_aGridDim.x;
	cudaMemset(m_gfSumPower, 0, iBytes);
	cudaMemset(m_gfCount, 0, iBytes);
	//-------------------------------
	int iShmBytes = 2 * m_aBlockDim.x * sizeof(float);
	//------------------------------------------------
	mGSumPower<<<m_aGridDim, m_aBlockDim, iShmBytes>>>
	( m_aiCmpSize[0], m_aiCmpSize[1], fStartFreq, fEndFreq,
	  m_gCmpImg, m_gfSumPower, m_gfCount
	);
	//----------------------------------
	float fPower = mSum1d(m_gfSumPower, m_aGridDim.x);
	float fCount = mSum1d(m_gfCount, m_aGridDim.x);
	fPower = (float)(fPower / (fCount + 1e-30));
	return fPower;
}

float GCorCTF2D::mSum1d(float* gfData, int iSize)
{
	if(iSize < 512)
	{	float fSum = mSum1dCPU(gfData, iSize);
		return fSum;
	}
	//------------------
	dim3 aGridDim(iSize / m_aBlockDim.x + 1, 1);
	int iBytes = sizeof(float) * aGridDim.x;
	float* gfSum = 0L;
	cudaMalloc(&gfSum, iBytes);
	cudaMemset(gfSum, 0, iBytes);
	//---------------------------
	int iShmBytes = sizeof(float) * m_aBlockDim.x;
	mGSum1D<<<aGridDim, m_aBlockDim, iShmBytes>>>
	( iSize, gfData, gfSum
	);
	//--------------------
	float fSum = mSum1dCPU(gfSum, aGridDim.x);
	cudaFree(gfSum);
	return fSum;
}

float GCorCTF2D::mSum1dCPU(float* gfData, int iSize)
{
	float* pfData = new float[iSize];
	int iBytes = sizeof(float) * iSize;
	cudaMemcpy(pfData, gfData,  iBytes, cudaMemcpyDefault);
	//-----------------------------------------------------
	float fSum = 0.0f;
	for(int i=0; i<iSize; i++)
	{	fSum += pfData[i];
	}
	//------------------------
	if(pfData != 0L) delete[] pfData;
	return fSum;
}
