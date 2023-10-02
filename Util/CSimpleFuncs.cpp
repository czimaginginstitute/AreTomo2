#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>

size_t Util::GetFloatBytes(int* piSize)
{
	size_t tBytes = sizeof(float) * piSize[0];
	tBytes *= piSize[1];
	return tBytes;
}

size_t Util::GetCmpBytes(int* piSize)
{
	size_t tBytes = sizeof(cufftComplex) * piSize[0];
	tBytes *= piSize[1];
	return tBytes;
}

size_t Util::GetUCharBytes(int* piSize)
{
	size_t tBytes = sizeof(char) * piSize[0];
	tBytes *= piSize[1];
	return tBytes;
}

float* Util::GGetFBuf(int* piSize, bool bZero)
{
	size_t tBytes = Util::GetFloatBytes(piSize);
	float* gfBuf = 0L;
	cudaMalloc(&gfBuf, tBytes);
	if(bZero) cudaMemset(gfBuf, 0, tBytes);
	return gfBuf;
}

cufftComplex* Util::GGetCmpBuf(int* piSize, bool bZero)
{
	size_t tBytes = Util::GetCmpBytes(piSize);
	cufftComplex* gCmpBuf = 0L;
	cudaMalloc(&gCmpBuf, tBytes);
	if(bZero) cudaMemset(gCmpBuf, 0, tBytes);
	return gCmpBuf;
}

unsigned char* Util::GGetUCharBuf(int* piSize, bool bZero)
{
	size_t tBytes = Util::GetUCharBytes(piSize);
	unsigned char* gucBuf = 0L;
	cudaMalloc(&gucBuf, tBytes);
	if(bZero) cudaMemset(gucBuf, 0, tBytes);
	return gucBuf;
}

void Util::PrintGpuMemoryUsage(const char* pcInfo)
{
	size_t tTotal = 0, tFree = 0;
	cudaError_t tErr = cudaMemGetInfo(&tFree, &tTotal);
	//-------------------------------------------------
	tTotal /= (1024 * 1024);
	tFree /= (1024 * 1024);
	printf("%s: %ld  %ld\n", pcInfo, tTotal, tFree);
}

void Util::CheckCudaError(const char* pcLocation)
{
	cudaError_t cuErr = cudaGetLastError();
	if(cuErr == cudaSuccess) return;
	//------------------------------
	fprintf(stderr, "%s:\n\t%s\n\n", pcLocation,
		cudaGetErrorString(cuErr));
	cudaDeviceReset();
	assert(0);
}
