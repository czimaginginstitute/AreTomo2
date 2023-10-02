#include "CMrcUtilInc.h"
#include <stdio.h>
#include <memory.h>

using namespace MrcUtil;

CSaveTomoMrc* CSaveTomoMrc::m_pInstance = 0L;

CSaveTomoMrc* CSaveTomoMrc::GetInstance(void)
{
	if(m_pInstance == 0L) m_pInstance = new CSaveTomoMrc;
	return m_pInstance;
}

void CSaveTomoMrc::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CSaveTomoMrc::CSaveTomoMrc(void)
{
	m_ppfImgs = 0L;
	m_ppfExts = 0L;
	m_fPixelSize = 0.0f;
	m_iNumFloats = 32;
}

CSaveTomoMrc::~CSaveTomoMrc(void)
{
	mClearImages();
	mClearExtHeaders();
}

bool CSaveTomoMrc::OpenFile(char* pcMrcFile)
{
	m_bOpen = m_aSaveMrc.OpenFile(pcMrcFile);
	return m_bOpen;
}

void CSaveTomoMrc::Start(int iNumImgs)
{	
	mClearImages();
	mClearExtHeaders();
	//-----------------
	m_aiImgSize[0] = 0;
	m_aiImgSize[1] = 0;
	m_aiImgSize[2] = iNumImgs;
	//------------------------
	m_aSaveMrc.SetMode(Mrc::eMrcFloat);
	m_aSaveMrc.SetImgSize(m_aiImgSize, m_aiImgSize[2]);
	m_aSaveMrc.SetExtHeader(0, m_iNumFloats, iNumImgs, 0);
	//----------------------------------------------------
	m_ppfImgs = new float*[iNumImgs];
	m_ppfExts = new float*[iNumImgs];
	memset(m_ppfImgs, 0, iNumImgs * sizeof(float*));
	memset(m_ppfExts, 0, iNumImgs * sizeof(float*));
	//----------------------------------------------
	Util_Thread::Start();
}

void CSaveTomoMrc::AsyncSave
(	float* pfImage,
	int* piImgSize,
   	float* pfExt,
   	int iElems,
   	int iImage
)
{	pthread_mutex_lock(&m_aMutex);
	m_ppfImgs[iImage] = pfImage;
	if(m_aiImgSize[0] == 0 || m_aiImgSize[1] == 0)
	{	m_aiImgSize[0] = piImgSize[0];
		m_aiImgSize[1] = piImgSize[1];
		m_aSaveMrc.SetImgSize(m_aiImgSize, m_aiImgSize[2]);
	}
	//---------------------------------------------------------
	float* pfBuf = new float[m_iNumFloats];
	int iSize = (iElems < m_iNumFloats) ? iElems : m_iNumFloats;
	int iBytes = iSize * sizeof(float);
	memcpy(pfBuf, pfExt, iBytes);
	if(m_ppfExts[iImage] != 0L) delete[] m_ppfExts[iImage];
	m_ppfExts[iImage] = pfBuf;
	//------------------------
	pthread_mutex_unlock(&m_aMutex);
}
	
void CSaveTomoMrc::ThreadMain(void)
{
	printf("\nSave Tomo MRC thread has been started.\n\n");
	int iSavedImgs = 0;
	//-----------------
	while(true)
	{	for(int i=0; i<m_aiImgSize[2]; i++)
		{	pthread_mutex_lock(&m_aMutex);
			float* pfImg = m_ppfImgs[i];
			float* pfExt = m_ppfExts[i];
			pthread_mutex_unlock(&m_aMutex);
			//------------------------------
			if(pfImg == 0L) continue;
			mSaveImage(i);
			iSavedImgs++;
		}
		if(iSavedImgs == m_aiImgSize[2]) break;
		mWait(1.0f);
	}
	m_aSaveMrc.CloseFile();
	mClearImages();
	mClearExtHeaders();
	printf("SaveMrcThread exits.\n\n");
}

void CSaveTomoMrc::mSaveImage(int iNthImg)
{
	float* pfImg = m_ppfImgs[iNthImg];
	float* pfExt = m_ppfExts[iNthImg];
	m_aSaveMrc.m_pSaveImg->DoIt(iNthImg, pfImg);
	//------------------------------------------
	if(pfExt != 0L) 
	{	int iBytes = m_iNumFloats * sizeof(float);
		m_aSaveMrc.m_pSaveExt->SetHeader((char*)pfExt, iBytes);
		m_aSaveMrc.m_pSaveExt->DoIt(iNthImg);
		m_fPixelSize = pfExt[11];
	}
	//------------------------------
	m_aSaveMrc.m_pSaveMain->SetPixelSize(m_fPixelSize);
	m_aSaveMrc.m_pSaveMain->DoIt();
	//-----------------------------
	if(pfImg != 0L) delete[] pfImg;
	if(pfExt != 0L) delete[] pfExt;
	m_ppfImgs[iNthImg] = 0L;
	m_ppfExts[iNthImg] = 0L;
	printf("\nImage %04d has been saved.\n\n", iNthImg+1);
}

void CSaveTomoMrc::mClearImages(void)
{
	if(m_ppfImgs == 0L) return;
	for(int i=0; i<m_aiImgSize[2]; i++)
	{	float* pfImg = m_ppfImgs[i];
		if(pfImg != 0L) delete[] pfImg;
	}
	delete[] m_ppfImgs;
	m_ppfImgs = 0L;
}

void CSaveTomoMrc::mClearExtHeaders(void)
{
	if(m_ppfExts == 0L) return;
	for(int i=0; i<m_aiImgSize[2]; i++)
	{	float* pfExt = m_ppfExts[i];
		if(pfExt != 0L) delete[] pfExt;
	}
	delete[] m_ppfExts;
	m_ppfExts = 0L;
}
