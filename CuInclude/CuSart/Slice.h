#pragma once

class CSlice
{
public:
	CSlice(void);
	~CSlice(void);
	void SetSize(int iWidth, int iHeight);
	int GetPixels(void);
	void SetZero(void);
	void SetData(int iX, int iY, float fVal);
	float GetData(int iX, int iY);
	void SetData(float* pfData);
	void Copy(CSlice* pSlice);
	CSlice* GetCopy(void);
	float* m_pfData;
	int m_iWidth;
	int m_iHeight;
private:
	void mAllocate(void);
	void mFree(void);
};
