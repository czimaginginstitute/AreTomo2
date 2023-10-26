#pragma once

#include <Util/Util_Volume.h>

class CTomoSmooth
{
public:
	CTomoSmooth(void);
	~CTomoSmooth(void);
	void SetKernel(Util_Volume* pKernel)
	{	m_pKernel = pKernel;
	}
	void SetDeviceInfo(int iNumDevices, int iDeviceID, int iThreadID)
	{	m_iNumDevices = iNumDevices;
		m_iDeviceID = iDeviceID;
		m_iThreadID = iThreadID;
	}
	void SetVolume(Util_Volume* pVolume)
	{	m_pVolume = pVolume;
	}
	void DoIt(void);
	void Assemble(void);
private:
	Util_Volume* m_pVolume;
	Util_Volume* m_pKernel;
	int m_iThreadID;	// must be zero based
	int m_iNumDevices;
	int m_iDeviceID;
	void* m_pvSplitVolumeX;
	Util_Volume* m_pSplitVol;
};
