#include "CCorrectInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace Correct;

void CCorrectUtil::CalcAlignedSize
(	int* piRawSize, float fTiltAxis,
	int* piAlnSize
)
{	memcpy(piAlnSize, piRawSize, sizeof(int) * 2);
	double dRot = fabs(sin(fTiltAxis * 3.14 / 180.0));
	if(dRot <= 0.707) return;
	//-----------------------
	piAlnSize[0] = piRawSize[1];
	piAlnSize[1] = piRawSize[0];		
}

void CCorrectUtil::CalcBinnedSize
(	int* piRawSize, float fBinning, bool bFourierCrop,
	int* piBinnedSize
)
{	if(bFourierCrop)
	{	Util::GFourierCrop2D::GetImgSize(piRawSize,
		   fBinning, piBinnedSize);
	}
	else
	{	int iBin = (int)(fBinning + 0.5f);
		bool bPadded = true;
		Util::GBinImage2D::GetBinSize(piRawSize, !bPadded,
		   iBin, piBinnedSize, !bPadded);
	}
}

