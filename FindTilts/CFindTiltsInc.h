#pragma once
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <Util/Util_Thread.h>
#include <Util/Util_Powell.h>
#include <cufft.h>

namespace FindTilts
{

class CTotalCC : public Util_Thread
{
public:
	CTotalCC(void);
	~CTotalCC(void);
	static float DoIt
	( float** ppfProjs,
	  float* pfTilts,
	  float* pfTiltAxes,
	  int* piProjSize,
	  int* piGpuIDs,
	  int iNumGpus
	);
	void Run(Util::CNextItem* pNextItem, int iThreadId);
	void ThreadMain(void);
private:
	Util::CStretchCC2D m_stretchCC;
	Util::CNextItem* m_pNextItem;
	float m_fCCSum;
	float m_fStdSum;
	int m_iGpuID;
};

class CFindTilts 
{
public:
	CFindTilts(void);
	~CFindTilts(void);
	void Clean(void);
	void Setup
	( int iIterations,
	  float fTol,
	  float fBFactor
	);
	void Find
	( MrcUtil::CTomoStack* pTomoStack,
	  float* pfTiltRange,
	  int* piGpuIds,
	  int iNumGpus
	);
private:
	void mFindTilts(void);
	float mCalcCC(int iStartProj, int iEndProj);
	void mCalcTilts(int iStartProj, int iEndProj);
	MrcUtil::CTomoStack* m_pTomoStack;
	int* m_piGpuIDs;
	int m_iNumGpus;
	float* m_pfTilts;
	float* m_pfTiltAxes;
	float m_afTiltRange[2];
	int m_iNumProjs;
};

}
