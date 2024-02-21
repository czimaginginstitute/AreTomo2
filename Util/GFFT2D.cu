#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>

using namespace Util;

static __global__ void mGMultiply
(	cufftComplex* gCmp, int iCmpY,
	float fFactor
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
	gCmp[i].x *= fFactor;
	gCmp[i].y *= fFactor;
}

static __global__ void mGRemoveAmp(cufftComplex* gCmp, int iCmpY)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	float fRe = gCmp[i].x;
	float fIm = gCmp[i].y;
	fRe = sqrtf(fRe * fRe + fIm * fIm) + (float)1e-20;
	gCmp[i].x /= fRe;
	gCmp[i].y /= fRe;
}

GFFT2D::GFFT2D(void)
{
	m_aStream = (cudaStream_t)0;
}

GFFT2D::~GFFT2D(void)
{
}

void GFFT2D::SetStream(cudaStream_t stream)
{
	m_aStream = stream;
}

void GFFT2D::DestroyPlan(void)
{
	if(m_cufftPlan == 0) return;
	cufftDestroy(m_cufftPlan);
	m_cufftPlan = 0;
	m_aiFFTSize[0] = 0;
	m_aiFFTSize[1] = 0;
	m_cufftType = CUFFT_R2C;
}

void GFFT2D::CreatePlan(int* piFFTSize, bool bForward)
{	
	cufftType fftType = bForward ? CUFFT_R2C : CUFFT_C2R;
	if(m_cufftType != fftType) this->DestroyPlan();
	else if(m_aiFFTSize[0] != piFFTSize[0]) this->DestroyPlan();
	else if(m_aiFFTSize[1] != piFFTSize[1]) this->DestroyPlan();
	if(m_cufftPlan != 0) return;
	//--------------------------
	m_cufftType = fftType;
	m_aiFFTSize[0] = piFFTSize[0];
	m_aiFFTSize[1] = piFFTSize[1];
	cufftResult res = cufftPlan2d(&m_cufftPlan, 
	   m_aiFFTSize[1], m_aiFFTSize[0], m_cufftType);
	mCheckError(res, "GFFT2D::CreatePlan");
}

void GFFT2D::Forward(float* gfPadImg, bool bNorm)
{
	cufftSetStream(m_cufftPlan, m_aStream);	
	cufftResult res = cufftExecR2C(m_cufftPlan, 
	   (cufftReal*)gfPadImg, (cufftComplex*)gfPadImg);
	if(bNorm) mNormalize((cufftComplex*)gfPadImg);
	mCheckError(res, "GFFT2D::Forward 1");
}

void GFFT2D::Forward(float* gfImg, cufftComplex* gCmp, bool bNorm)
{
	cufftSetStream(m_cufftPlan, m_aStream);
	cufftResult res = cufftExecR2C(m_cufftPlan,
	   (cufftReal*)gfImg, gCmp);
	if(bNorm) mNormalize(gCmp);
	mCheckError(res, "GFFT2D::Forward 2");
}

void GFFT2D::Inverse(cufftComplex* gCmp)
{
	cufftSetStream(m_cufftPlan, m_aStream);	
	cufftResult res = cufftExecC2R(m_cufftPlan, gCmp, 
	   (cufftReal*)gCmp);
	mCheckError(res, "GFFT2D::Inverse 1");
}

void GFFT2D::Inverse(cufftComplex* gCmpImg, float* gfImg)
{
	cufftSetStream(m_cufftPlan, m_aStream);
	cufftResult res = cufftExecC2R(m_cufftPlan, gCmpImg,
	   (cufftReal*)gfImg);
	mCheckError(res, "GFFT2D::Inverse 2");
}

void GFFT2D::RemoveAmp(cufftComplex* gCmp, int* piCmpSize)
{
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piCmpSize[0], piCmpSize[1] / aBlockDim.y + 1);
	mGRemoveAmp<<<aGridDim, aBlockDim, 0, m_aStream>>>(gCmp, piCmpSize[1]);
}

void GFFT2D::mNormalize(cufftComplex* gCmpImg)
{
	int iCmpSizeX = m_aiFFTSize[0] / 2 + 1;
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iCmpSizeX, 1);
	aGridDim.y = (m_aiFFTSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	float fFactor = (1.0f / m_aiFFTSize[0]) / m_aiFFTSize[1];
	mGMultiply<<<aGridDim, aBlockDim, 0, m_aStream>>>(gCmpImg, 
	   m_aiFFTSize[1], fFactor);
}

void GFFT2D::mCheckError(cufftResult error, const char* pcFunc)
{
	switch(error)
	{	case CUFFT_SUCCESS: 
		return;
		//-----------------
		case CUFFT_INVALID_PLAN:
		printf("%s: CUFFT_INVALID_PLAN\n\n", pcFunc);
		//-------------------------------------------
		case CUFFT_ALLOC_FAILED: 
		printf("%s: CUFFT_ALLOC_FAILED\n\n", pcFunc);
		//-------------------------------------------
		case CUFFT_INVALID_TYPE:
                printf("%s: CUFFT_INVALID_TYPE\n\n", pcFunc);
                //-------------------------------------------
		case CUFFT_INVALID_VALUE:
		printf("%s: CUFFT_INVALID_VALUE\n\n", pcFunc);
                //--------------------------------------------
		case CUFFT_INTERNAL_ERROR:
		printf("%s: CUFFT_INTERNAL_ERROR\n\n", pcFunc);
		//---------------------------------------------
		case CUFFT_EXEC_FAILED:
		printf("%s: CUFFT_EXEC_FAILED\n\n", pcFunc);
		//------------------------------------------
		case CUFFT_SETUP_FAILED:
		printf("%s: CUFFT_SETUP_FAILED\n\n", pcFunc);
		//------------------------------------------
		case CUFFT_INVALID_SIZE:
		printf("%s: CUFFT_INVALID_SIZE\n\n", pcFunc);
		//-------------------------------------------
		case CUFFT_UNALIGNED_DATA:
		printf("%s: CUFFT_UNALIGNED_DATA\n\n", pcFunc);
        }
}
