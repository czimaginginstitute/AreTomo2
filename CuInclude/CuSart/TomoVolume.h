#pragma once

#include <Util/Util_Volume.h>
#include <pthread.h>

//===================================================================
// This is a synchronized version of Util_Volume to be reconstructed 
// on a host equipped with multiple GPUs.
//===================================================================
class CTomoVolume
{
public:
	CTomoVolume(void);
	CTomoVolume(Util_Volume* pVolume, int iNumDevices);
	~CTomoVolume(void);
	void Set(Util_Volume* pVolume, int iNumDevices);
	// -----------------------
	// Wait all GPUs to finish
	// -----------------------
	void WaitAllDone(void);
	// ------------------------------------
	// All any GPU to notify its completion
	// ------------------------------------
	void NotifyDone(void);
	Util_Volume* m_pVolume;	// do not free
	int* m_piSize;
private:
	pthread_cond_t m_aCond;
	pthread_mutex_t m_aMutex;
	int m_iNumDevices;
	int m_iDoneDevices;
	bool m_bAllDone;
};
