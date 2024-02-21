#include "CInput.h"
#include "Util/CUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

CInput* CInput::m_pInstance = 0L;


CInput* CInput::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CInput;
	return m_pInstance;
}

void CInput::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CInput::CInput(void)
{
	strcpy(m_acInMrcTag, "-InMrc");
	strcpy(m_acOutMrcTag, "-OutMrc");
	strcpy(m_acAlnFileTag, "-AlnFile");
	strcpy(m_acAngFileTag, "-AngFile");
	strcpy(m_acRoiFileTag, "-RoiFile");
	strcpy(m_acTmpFileTag, "-TmpFile");
	strcpy(m_acLogFileTag, "-LogFile");
	strcpy(m_acTiltRangeTag, "-TiltRange");
	strcpy(m_acTiltAxisTag, "-TiltAxis");
	strcpy(m_acAlignZTag, "-AlignZ");
	strcpy(m_acVolZTag, "-VolZ");
	strcpy(m_acOutBinTag, "-OutBin");
	strcpy(m_acTiltCorTag, "-TiltCor");
	strcpy(m_acGpuIDTag, "-Gpu");
	strcpy(m_acReconRangeTag, "-ReconRange");
	strcpy(m_acPixelSizeTag, "-PixSize");
	strcpy(m_acKvTag, "-Kv");
	strcpy(m_acImgDoseTag, "-ImgDose");
	strcpy(m_acCsTag, "-Cs");
	strcpy(m_acAmpContrastTag, "-AmpContrast");
	strcpy(m_acExtPhaseTag, "-ExtPhase");
	strcpy(m_acFlipVolTag, "-FlipVol");
	strcpy(m_acFlipIntTag, "-FlipInt");
	strcpy(m_acSartTag, "-Sart");
	strcpy(m_acWbpTag, "-Wbp");
	strcpy(m_acPatchTag, "-Patch");
	strcpy(m_acTiltSchemeTag, "-TiltScheme");
	strcpy(m_acOutXFTag, "-OutXF");
	strcpy(m_acAlignTag, "-Align");
	strcpy(m_acCropVolTag, "-CropVol");
	strcpy(m_acOutImodTag, "-OutImod");
	strcpy(m_acDarkTolTag, "-DarkTol");
	strcpy(m_acBFactorTag, "-Bft");
	strcpy(m_acIntpCorTag, "-IntpCor");
	//---------------------------------
	m_afTiltRange[0] = 0.0f;
	m_afTiltRange[1] = 0.0f;
	m_afTiltAxis[0] = 0.0f;
	m_afTiltAxis[1] = 1.0f;
	m_iAlignZ = 600;
	m_iVolZ = 1200;
	m_fOutBin = 1.0f;
	m_piGpuIDs = 0L;
	m_iNumGpus = 0;
	m_afTiltCor[0] = 0.0f;
	m_afTiltCor[1] = 0.0f;
	m_afReconRange[0] = -90.0f;
	m_afReconRange[1] = 90.0f;
	m_fPixelSize = 0.0f;
	m_fKv = 0.0f;
	m_fCs = 0.0f;
	m_fAmpContrast = 0.07f;
	m_iFlipVol = 0;
	m_iFlipInt = 0;
	m_aiSartParam[0] = 20;
	m_aiSartParam[1] = 5;
	m_iWbp = 0;
	m_iOutXF = 0;
	m_iAlign = 1;
	m_fDarkTol = 0.7f;
	m_afBFactor[0] = 500.0f;
	m_afBFactor[1] = 500.0f;
	m_iIntpCor = 1;
	memset(m_afExtPhase, 0, sizeof(m_afExtPhase));
	memset(m_aiNumPatches, 0, sizeof(m_aiNumPatches));
	memset(m_aiCropVol, 0, sizeof(m_aiCropVol));
}

CInput::~CInput(void)
{
	if(m_piGpuIDs != 0L) delete[] m_piGpuIDs;
}

void CInput::ShowTags(void)
{
	printf("%-10s\n"
	"  1. Input MRC file that stores tomo tilt series.\n\n",
	m_acInMrcTag);
	//------------
        printf("%-10s\n"
	"  1. Output MRC file that stores the aligned tilt series.\n\n",
	m_acOutMrcTag);
	//-------------
	printf("%-10s\n"
	"  1. Alignment file to be loaded.\n"
	"  2. It will be applied to the loaded tilt series.\n\n",
	m_acAlnFileTag);
	//--------------
	printf("%-10s\n"
	"  1. A single- or multi-column Text file that contains tilt\n"
	"     angles in the first column.\n"
	"  2. Both the number and the order of tilt angles must match\n"
	"     the number and order of projection images in the input\n"
	"     MRC file.\n\n", m_acAngFileTag);
	//------------------------------------ 
        printf("%-10s\n"
	"  1. Temporary image file for debugging.\n\n",
	m_acTmpFileTag);
	//--------------
	printf("%-10s\n"
	"  1. Log file storing alignment data.\n\n",
	m_acLogFileTag);
	//--------------
	printf("%-10s\n", m_acTiltRangeTag);
	printf("   Min and max tilts. By default the header values ");
	printf("are used.\n\n");
	//----------------------
	printf("%-10s\n", m_acTiltAxisTag);
	printf("   Tilt axis, default header value.\n\n");
	//------------------------------------------------
	printf("%-10s\n", m_acAlignZTag);
	printf("   Volume height for alignment, default 256\n\n");
	//--------------------------------------------------------
	printf("%-10s\n", m_acVolZTag);
	printf("   1. Volume z height for reconstrunction. It must be\n");
	printf("      greater than 0 to reconstruct a volume.\n");
	printf("   2. Default is 0, only aligned tilt series will\n");
	printf("      generated.\n\n");
	//-----------------------------
	printf("%-10s\n", m_acOutBinTag);
	printf("   Binning for aligned output tilt series, default 1\n\n");
	//-----------------------------------------------------------------
        printf("%-10s\n", m_acGpuIDTag);
	printf("   GPU IDs. Default 0.\n\n");
	//-----------------------------------
	printf("%-10s\n", m_acTiltCorTag);
        printf("   1. Correct the offset of tilt angle.\n");
        printf("   2. This argument can be followed by two values. The\n"
           "      first value can be -1, 0, or 1. and the  default is 0,\n"
	   "      indicating the tilt offset is measured for alignment\n"
	   "      only  When the value is 1, the offset is applied to\n"
	   "      reconstion too. When a negative value is given, tilt\n"
	   "      is not measured not applied.\n"
           "   3. The second value is user provided tilt offset. When it\n"
	   "      is given, the measurement is disabled.\n\n");
	//-----------------------------------------------------
	printf("%-10s\n", m_acReconRangeTag);
	printf("   1. It specifies the min and max tilt angles from which\n");
	printf("      a 3D volume will be reconstructed. Any tilt image\n");
	printf("      whose tilt ange is outside this range is exclueded\n");
	printf("      in the reconstruction.\n\n");
	//-----------------------------------------
	printf("%-10s\n", m_acPixelSizeTag);
	printf("   1. Pixel size in Angstrom of the input tilt series. It\n"
	   "      is only required for dose weighting. If missing, dose\n"
	   "      weighting will be disabled.\n\n");
	//------------------------------------------
	printf("%-10s\n", m_acKvTag);
	printf("   1. High tension in kV\n");
	printf("   2. Required for dose weighting and CTF estimation\n\n");
	//-----------------------------------------------------------------
	printf("%-10s\n", m_acImgDoseTag);
	printf("   1. Dose on sample in each image exposure in e/A2. Note\n"
	   "      this is not accumulated dose. If missing, dose weighting\n"
	   "      will be disabled.\n\n");
	//--------------------------------
	printf("%-10s\n", m_acCsTag);
	printf("   1. Spherical aberration in mm\n");
	printf("   2. Requred only for CTF correction\n\n");
	printf("$-10s\n", m_acAmpContrastTag);
	printf("   1. Amplitude contrast, default 0.07\n\n");
	printf("-10s\n", m_acExtPhaseTag);
	printf("   1. Guess of phase shift and search range in degree.\n");
	printf("   2. Only required for CTF estimation and with\n");
	printf("   3. Phase plate installed.\n\n");
	//-----------------------------------------
	printf("%-10s\n", m_acFlipVolTag);
	printf("   1. By giving a non-zero value, the reconstructed\n");
	printf("      volume is saved in xyz fashion. The default is\n");
	printf("      xzy.\n");
	//---------------------
	printf("%-10s\n"
	"  1. Flip the intensity of the volume to make structure white.\n"
	"     Default 0 means no flipping. Non-zero value flips.\n",
	m_acFlipIntTag);
	//--------------
	printf("%-10s\n", m_acSartTag);
	printf("   1. Specify number of SART iterations and number\n");
	printf("      of projections per update. The default values\n");
	printf("      are 15 and 5, respectively\n\n");
	//---------------------------------------------
	printf("%-10s\n", m_acWbpTag);
	printf("   1. By specifying 1, weighted back projection is enabled\n");
	printf("      to reconstruct volume.\n\n");
	//-----------------------------------------
	printf("%-10s\n", m_acDarkTolTag);
	printf("   1. Set tolerance for removing dark images. The range is\n"
	   "      in (0, 1). The default value is 0.7. The higher value is\n"
	   "      more restrictive.\n\n");
	//--------------------------------
	printf("%-10s\n", m_acTiltSchemeTag);
	printf("   1. This option is used to determine the sequence each\n"
	   "      tilt image is acquired. This sequence is needed for the\n"
	   "      determination of accumulated dose on sample. If this\n"
	   "      option is missing, dose weighting will be disabled.\n"
	   "   2. Three parameters are needed. This first one is the\n"
	   "      starting angle. The second, tilt step, positive or\n"
	   "      negative,  indicates tilting direction direction\n"
	   "      after the starting angle. The third is 1, 2, or 3,\n"
	   "      corresponding\n to single-branch, two-branch, or Hagen\n"
	   "      scheme of data collection, respectively.\n\n");
	//-------------------------------------------------------
	printf("%-10s\n", m_acOutXFTag);
	printf("   1. When set by giving no-zero value, IMOD compatible\n"
	   "      XF file will be generated.\n\n");
	//-----------------------------------------
	printf("%-10s\n", m_acOutImodTag);
	printf("   1. It generates the Imod files needed by Relion4 or Warp\n"
	   "      for subtomogram averaging. These files are saved in the\n"
	   "      subfolder named after the output MRC file name.\n"
	   "   2. 0: default, do not generate any IMod files.\n"
	   "   3. 1: generate IMod files needed for Relion 4.\n"
	   "   4. 2: generate IMod files needed for WARP.\n"
	   "   5. 3: generate IMod files when the aligned tilt series\n"
	   "         is used as the input for Relion 4 or WARP.\n\n");
	//------------------------------------------------------------
	printf("%-10s\n", m_acAlignTag);
	printf("   1. Skip alignment when followed by 0. This option is\n"
	   "      used when the input MRC file is an aligned tilt series.\n"
	   "      The default value is 1.\n\n");
	//--------------------------------------
	printf("%-10s\n", m_acCropVolTag);
	printf("   1. Crop the reconstructed volume to the specified sizes\n"
	   "      in x and y directions.\n"
	   "   2. Size x is the length perpendicular to tilt axis and size\n"
	   "      y is the length along the tilt axis.\n"
	   "   3. This option is only enabled when -RoiFile is enabled.\n\n");
	//--------------------------------------------------------------------
	printf("%-10s\n", m_acBFactorTag);
	printf("   1. B-factors for low-pass filter used in the cross\n"
	   "      correlation. The first value is used for global\n"
	   "      measurement. The second for the local measurement.\n\n");
	//-----------------------------------------------------------------
	printf("%-10s\n", m_acIntpCorTag);
	printf("   1. When enabled, the correction for information loss due\n"
	   "      to linear interpolation will be perform. The default\n"
	   "      setting value 1 enables the correction.\n\n");
}

void CInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//------------
	memset(m_acInMrcFile, 0, sizeof(m_acInMrcFile));
	memset(m_acOutMrcFile, 0, sizeof(m_acOutMrcFile));
	memset(m_acAlnFile, 0, sizeof(m_acAlnFile));
	memset(m_acAngFile, 0, sizeof(m_acAngFile));
	memset(m_acRoiFile, 0, sizeof(m_acRoiFile));
	memset(m_acTmpFile, 0, sizeof(m_acTmpFile));
	memset(m_acLogFile, 0, sizeof(m_acLogFile));
	//------------------------------------------
	int aiRange[2];
	Util::CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	aParseArgs.FindVals(m_acInMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInMrcFile);
	//-------------------------------------------
	aParseArgs.FindVals(m_acOutMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acOutMrcFile);
	//--------------------------------------------
	aParseArgs.FindVals(m_acAlnFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acAlnFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acAngFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acAngFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acRoiFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acRoiFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acTmpFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acTmpFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acLogFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acLogFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acTiltRangeTag, aiRange);
	if(aiRange[1] >= 2)
	{	aiRange[1] = 2;
		aParseArgs.GetVals(aiRange, m_afTiltRange);
	}
	//-------------------------------------------------
	aParseArgs.FindVals(m_acTiltAxisTag, aiRange);
	if(aiRange[1] >= 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afTiltAxis);
	//----------------------------------------
	aParseArgs.FindVals(m_acAlignZTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.FindVals(m_acAlignZTag, aiRange);
	aParseArgs.GetVals(aiRange, &m_iAlignZ);
	//--------------------------------------
	aParseArgs.FindVals(m_acVolZTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iVolZ);
	//------------------------------------
	aParseArgs.FindVals(m_acOutBinTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fOutBin);
	if(m_fOutBin < 1) m_fOutBin = 1;
	//------------------------------
	aParseArgs.FindVals(m_acTiltCorTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afTiltCor);	
	//---------------------------------------
	if(m_piGpuIDs != 0L) delete[] m_piGpuIDs;
	aParseArgs.FindVals(m_acGpuIDTag, aiRange);
	if(aiRange[1] >= 1)
	{	m_iNumGpus = aiRange[1];
		m_piGpuIDs = new int[m_iNumGpus];
		aParseArgs.GetVals(aiRange, m_piGpuIDs);
	}
	else
	{	m_iNumGpus = 1;
		m_piGpuIDs = new int[m_iNumGpus];
		m_piGpuIDs[0] = 0;
	}
	//------------------------
	aParseArgs.FindVals(m_acReconRangeTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afReconRange);
	//------------------------------------------
	aParseArgs.FindVals(m_acPixelSizeTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fPixelSize);
	//-----------------------------------------
	aParseArgs.FindVals(m_acKvTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fKv);
	//----------------------------------
	aParseArgs.FindVals(m_acCsTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fCs);
	//----------------------------------
	aParseArgs.FindVals(m_acAmpContrastTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fAmpContrast);
	//-------------------------------------------
	aParseArgs.FindVals(m_acExtPhaseTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afExtPhase);
	//----------------------------------------
	aParseArgs.FindVals(m_acImgDoseTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fImgDose);
	//---------------------------------------
	aParseArgs.FindVals(m_acFlipVolTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iFlipVol);
	//---------------------------------------
	aParseArgs.FindVals(m_acFlipIntTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iFlipInt);
	//---------------------------------------
	aParseArgs.FindVals(m_acSartTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiSartParam);
	if(m_aiSartParam[0] <= 0) m_aiSartParam[0] = 15;
	if(m_aiSartParam[1] < 1) m_aiSartParam[1] = 5;
	//--------------------------------------------
	aParseArgs.FindVals(m_acWbpTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iWbp);
	//-----------------------------------
	aParseArgs.FindVals(m_acPatchTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiNumPatches);
	//------------------------------------------
	aParseArgs.FindVals(m_acTiltSchemeTag, aiRange);
	if(aiRange[1] > 3) aiRange[1] = 3;
	aParseArgs.GetVals(aiRange, m_afTiltScheme);
	//------------------------------------------
	aParseArgs.FindVals(m_acOutXFTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iOutXF);
	//-------------------------------------
	aParseArgs.FindVals(m_acAlignTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iAlign);
	//-------------------------------------
	aParseArgs.FindVals(m_acCropVolTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiCropVol);
	//---------------------------------------
	aParseArgs.FindVals(m_acOutImodTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iOutImod);
	//---------------------------------------
	aParseArgs.FindVals(m_acDarkTolTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fDarkTol);
	//---------------------------------------
	aParseArgs.FindVals(m_acBFactorTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afBFactor);
	//---------------------------------------
	aParseArgs.FindVals(m_acIntpCorTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iIntpCor);
	//---------------------------------------
	if(strlen(m_acAlnFile) != 0) m_iAlign = 0;
	mPrint();	
}

void CInput::mPrint(void)
{
	printf("\n");
	printf("%-10s  %s\n", m_acInMrcTag, m_acInMrcFile);
	printf("%-10s  %s\n", m_acOutMrcTag, m_acOutMrcFile);
	printf("%-10s  %s\n", m_acAlnFileTag, m_acAlnFile);
	printf("%-10s  %s\n", m_acAngFileTag, m_acAngFile);
	printf("%-10s  %s\n", m_acRoiFileTag, m_acRoiFile);
	printf("%-10s  %s\n", m_acTmpFileTag, m_acTmpFile);
	printf("%-10s  %s\n", m_acLogFileTag, m_acLogFile);
	printf("%-10s  %d\n", m_acAlignZTag, m_iAlignZ);
	printf("%-10s  %d\n", m_acVolZTag, m_iVolZ);
	printf("%-10s  %.2f\n", m_acOutBinTag, m_fOutBin);
	if(m_afTiltRange[0] != 999)
	{	printf("%-10s  %.2f  %.2f\n", m_acTiltRangeTag,
		   m_afTiltRange[0], m_afTiltRange[1]);
	}
	printf("%-10s  %.2f  %.2f\n", m_acTiltAxisTag, 
		m_afTiltAxis[0], m_afTiltAxis[1]);
	printf("%-10s", m_acGpuIDTag);
	for(int i=0; i<m_iNumGpus; i++) 
	{	printf("  %d", m_piGpuIDs[i]);
	}
	printf("\n");
	printf("%-10s  %.2f  %.2f\n", m_acTiltCorTag, m_afTiltCor[0],
	   m_afTiltCor[1]);
	printf( "%-10s  %.2f  %.2f\n", m_acReconRangeTag, 
	   m_afReconRange[0], m_afReconRange[1]);
	//---------------------------------------
	printf("%-10s  %.2f\n", m_acPixelSizeTag, m_fPixelSize);
	printf("%-10s  %.2f\n", m_acImgDoseTag, m_fImgDose);
	printf("%-10s  %.2f\n", m_acKvTag, m_fKv);
	//----------------------------------------
	printf("%-10s  %.2f\n", m_acCsTag, m_fCs);
	printf("%-10s  %.2f\n", m_acAmpContrastTag, m_fAmpContrast);
	printf("%-10s  %.2f  %.2f\n", m_acExtPhaseTag, m_afExtPhase[0],
	   m_afExtPhase[1]);
	//-------------------------------------------------------------
	printf("%-10s  %d\n", m_acFlipVolTag, m_iFlipVol);
	printf("%-10s  %d\n", m_acFlipIntTag, m_iFlipInt);
	printf("%-10s  %d  %d\n", m_acSartTag, 
	   m_aiSartParam[0], m_aiSartParam[1]);
	printf("%-10s  %d\n", m_acWbpTag, m_iWbp);
	printf("%-10s  %d  %d\n", m_acPatchTag, m_aiNumPatches[0],
	   m_aiNumPatches[1]);
	printf("%-10s  %.2f  %.1f  %.1f\n", m_acTiltSchemeTag,
	   m_afTiltScheme[0], m_afTiltScheme[1], m_afTiltScheme[2]);
	printf("%-10s  %d\n", m_acOutXFTag, m_iOutXF);
	printf("%-10s  %d\n", m_acAlignTag, m_iAlign);
	printf("%-10s  %d  %d\n", m_acCropVolTag, m_aiCropVol[0],
	   m_aiCropVol[1]);
	printf("%-10s  %d\n", m_acOutImodTag, m_iOutImod);
	printf("%-10s  %.2f\n", m_acDarkTolTag, m_fDarkTol);
	printf("%-10s  %.1f  %.1f\n", m_acBFactorTag,
	   m_afBFactor[0], m_afBFactor[1]);
	printf("%-10s  %d\n", m_acIntpCorTag, m_iIntpCor);
	printf("\n");
}

char* CInput::GetLogFile(char* pcSuffix, int* piSerial)
{
	if(strlen(m_acLogFile) == 0) return 0L;
	char* pcLogFile = mGenFileName(m_acLogFile, pcSuffix, piSerial);
	return pcLogFile;
}

char* CInput::GetTmpFile(char* pcSuffix, int* piSerial)
{
	if(strlen(m_acTmpFile) == 0) return 0L;
	char* pcLogFile = mGenFileName(m_acLogFile, pcSuffix, piSerial);
	return pcLogFile;
}

float CInput::GetOutPixSize(void)
{
	float fPixSize = m_fPixelSize * m_fOutBin;
	if(fPixSize <= 0) return 0.0f;
	else return fPixSize;
}

char* CInput::mGenFileName
(	char* pcPrefix,
	char* pcSuffix, 
	int* piSerial
)
{	char* pcLogFile = new char[256];
	strcpy(pcLogFile, pcPrefix);
	//--------------------------
	char* pcExt = strchr(pcLogFile, '.');
	if(pcSuffix != 0L)
	{	if(pcExt == 0L) strcat(pcLogFile, pcSuffix);
		else strcpy(pcExt, pcSuffix);
	}
	//-----------------------------------	
	char acSerial[16] = {'\0'};
	if(piSerial != 0L) 
	{	sprintf(acSerial, "_%d", *piSerial);
		pcExt = strchr(pcLogFile, '.');
		if(pcExt == 0L) strcat(pcLogFile, acSerial);
		else strcpy(pcExt, acSerial);
	}	
	//-----------------------------------
	pcExt = strchr(pcLogFile, '.');
	if(pcExt != 0L) return pcLogFile;
	//-------------------------------
	pcExt = strchr(pcSuffix, '.');
	if(pcExt != 0L)
	{	strcat(pcLogFile, pcExt);
		return pcLogFile;
	}
	//-----------------------
	pcExt = strchr(pcPrefix, '.');
	if(pcExt != 0L)
	{	strcat(pcLogFile, pcExt);
		return pcLogFile;
	}
	//-----------------------
	return pcLogFile;
}

