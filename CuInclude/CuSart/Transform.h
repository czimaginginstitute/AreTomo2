#pragma once
class CTransform
{
public:
	CTransform(void);
	~CTransform(void);
	void SetTranMatrix(float* pfMatrix, int iSize);
	void SetIntScale(float* pfScale, int iSize);
	void SetMProj(float* pfProj, int iProjX, int iProjY);
	void SetAProj(float* pfProj, int iProjX, int iProjY);
	//--------------------------------------------------------------
	// This is the range that gives at each line of y the starting 
	// (inclusive) and ending (exclusive) x location in which 
	// valid data exist.
	//-------------------------------------------------------------- 
	void SetRangeX(unsigned int* puiRangeX);
	//--------------------------------------------------------------
	// This is the range that gives the starting y location and the
	// size of y in which the computation will be performed.
	//-------------------------------------------------------------- 
	void SetRangeY(int iStartY, int iSizeY);
	void DoIt(bool bLog);
private:
	void mUpdateRangeX(float* pfMiss);
	void* m_pvTransformImpl;
	float* m_pfAProj;
	int m_iAProjSize[2];
	int m_iRangeY[2];
	unsigned int* m_puiRangeX;
};
