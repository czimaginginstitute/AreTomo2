#pragma once
class CCompare
{
public:
	CCompare(void);
	~CCompare(void);
	void SetTranMatrix(float* pfMatrix, int iSize);
	void SetIntScale(float* pfScale, int iSize);
	void SetMProj(float* pfProj, int iProjX, int iProjY);
	void SetAProj(float* pfProj, int iProjX, int iProjY);
	void SetRangeY(int iStartY, int iSizeY);
	void DoIt(bool bLog);
	float m_fCount;
	float m_fSumDiff;
	float m_fSumCross;
	float m_fSumAligned;
	float m_fSumAlignedAbs;
	float m_fSumAlignedSqr;
	float m_fSumTransed;
	float m_fSumTransedAbs;
	float m_fSumTransedSqr;
private:
	void mCalcExecDim(void);
	void* m_pvCompareImpl;
};
