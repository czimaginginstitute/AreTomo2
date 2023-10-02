#include "CFindCTFInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace GCTFFind;

CCTFParam::CCTFParam(void)
{
}

CCTFParam::~CCTFParam(void)
{
}

float CCTFParam::CalcWavelength(void)
{
	float fKv = m_fHT * 1000.0f;
	double dLambda = 12.26 / sqrt(fKv * 1000.0 + 0.9784 * fKv * fKv);
	return (float)dLambda;
}

