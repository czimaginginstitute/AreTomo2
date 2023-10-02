#include "CMrcUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace MrcUtil;

CGenCentralSlices::CGenCentralSlices(void)
{
	m_pfSliceXY = 0L; // projection onto xy plane
	m_pfSliceYZ = 0L; // projection onto yz plane
	m_pfSliceXZ = 0L; // projection onto xz plane
}

CGenCentralSlices::~CGenCentralSlices(void)
{
	this->Clean();
}

void CGenCentralSlices::Clean(void)
{
	if(m_pfSliceXY != 0L) delete[] m_pfSliceXY;
	if(m_pfSliceYZ != 0L) delete[] m_pfSliceYZ;
	if(m_pfSliceXZ != 0L) delete[] m_pfSliceXZ;
	m_pfSliceXY = 0L;
	m_pfSliceYZ = 0L;
	m_pfSliceXZ = 0L; 
}

//-------------------------------------------------------------------
// Note: pVolStack is arranged in xzy order where x is fastest and
//       y is slowest index. In MRC file each section is a xz slice.
//-------------------------------------------------------------------
void CGenCentralSlices::DoIt(CTomoStack* pVolStack)
{
	this->Clean();
	m_iSizeX = pVolStack->m_aiStkSize[0];
	m_iSizeY = pVolStack->m_aiStkSize[2];
	m_iSizeZ = pVolStack->m_aiStkSize[1];
	//-----------------------------------
	mIntegrateX(pVolStack);
	mIntegrateY(pVolStack);
	mIntegrateZ(pVolStack);
	printf("Done with computing orthogonal projections.\n\n");
}

void CGenCentralSlices::GetSizeXY(int* piSize)
{
	piSize[0] = m_iSizeX;
	piSize[1] = m_iSizeY;
}

void CGenCentralSlices::GetSizeYZ(int* piSize)
{
	piSize[0] = m_iSizeY;
	piSize[1] = m_iSizeZ;
}

void CGenCentralSlices::GetSizeXZ(int* piSize)
{
	piSize[0] = m_iSizeX;
	piSize[1] = m_iSizeZ;
}

void CGenCentralSlices::mIntegrateX(CTomoStack* pVolStack)
{
	printf("Calculate yz projection.\n");
	int iPixels = m_iSizeY * m_iSizeZ;
	m_pfSliceYZ = new float[iPixels];
	memset(m_pfSliceYZ, 0, sizeof(float) * iPixels);
	//----------------------------------------------
	for(int y=0; y<pVolStack->m_aiStkSize[2]; y++)
	{	float* pfFrameXZ = pVolStack->GetFrame(y);
		for(int z=0; z<pVolStack->m_aiStkSize[1]; z++)
		{	float fSum = 0.0f;
			int iOffset = z * m_iSizeX;
			for(int x=0; x<pVolStack->m_aiStkSize[0]; x++)
			{	fSum += pfFrameXZ[iOffset+x];
			}
			m_pfSliceYZ[z * m_iSizeY + y] = fSum;	
		}
	}
}

void CGenCentralSlices::mIntegrateY(CTomoStack* pVolStack)
{
	printf("Calculate xz projection.\n");
	int iPixels = m_iSizeX * m_iSizeZ;
	m_pfSliceXZ = new float[iPixels];
	memset(m_pfSliceXZ, 0, sizeof(float) * iPixels);
	//----------------------------------------------
	for(int y=0; y<pVolStack->m_aiStkSize[2]; y++)
	{	float* pfFrameXZ = pVolStack->GetFrame(y);
		for(int i=0; i<iPixels; i++)
		{	m_pfSliceXZ[i] += pfFrameXZ[i];
		}
	}
}

void CGenCentralSlices::mIntegrateZ(CTomoStack* pVolStack)
{
	printf("Calculate xy projection.\n");
	int iPixels = m_iSizeX * m_iSizeY;
	m_pfSliceXY = new float[iPixels];
	memset(m_pfSliceXY, 0, sizeof(float) * iPixels);
	//----------------------------------------------
	for(int y=0; y<pVolStack->m_aiStkSize[2]; y++)
	{	float* pfFrameXZ = pVolStack->GetFrame(y);
		for(int x=0; x<pVolStack->m_aiStkSize[0]; x++)
		{	float fSum = 0.0f;
			for(int z=0; z<pVolStack->m_aiStkSize[1]; z++)
			{	fSum += pfFrameXZ[z * m_iSizeX + x];
			}
			m_pfSliceXY[y * m_iSizeX + x] = fSum;
		}
	}
}
