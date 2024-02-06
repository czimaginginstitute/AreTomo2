Version 1.0.2
-------------
1. Fixed bug in CCommonLineMain.cpp: When the initial tilt axis is less than
   -45 deg, add 180 to make it possitive.
2. Fixed bug in MrcUtil/CTiltDoses.cpp that could not the second column in
   the input angle file.
3. Add a second parameter for -TiltCor. It is the user provided tilt offset.
   When it appears, the measurement of tilt offset will be skipped.

Version 1.0.3
-------------
1. Changed Recon/GRWeight.cu used for weighted back projection. It now uses
   ramp filter plus hamming window.
2. Add -Align 0 that allows users to reconstruct aligned tilt series without
   running alignment.
3. When users only want aligned tilt series, an angle file is also created
   with the same name output MRC file but ended with ".tlt".
4. Now FindTiltOffset in CProcessThread is called only once.
5. Disabled -TiltScheme. For dose weighting, users need to add a second
   column 

Version 1.0.4
-------------
1. Generate three projections (xy, yz, xz) of reconstructed volumes and
   save them in MRC files whose names share the volume file name but
   appended with _projxy, _projyz, _projxz. These file are generated
   only when the volume is reconstructed.

Version 1.0.5
-------------
1. Revised PatchAlign/CFitPatchShifts.cpp that implemented the refinement
   of tilt axis for each projection based upon all local measurements
   throughout the tilt series.
2. Add -CropVol size_x size_y. This option crops the full volume to
   have the specified x and y sized and leave the z size untouched.
   The cropped volume will be centered at the geometric center of
   the targers in the target file following -RoiFile.
3. Revised /Recon/GRWeight.cu. Ramp filter plua Hamming window.

Version 1.0.6
-------------
1. Added bSart in Recon/GBackProj.cu for SART reconstruction.
   CTomoSart.cpp and CTomoWbp.cpp are changed correspondingly.
2. Revised ProjAlign/GReproj.cu. All projections are involved in the
   calculation of back projection but weighted based upon angular
   difference. When the deference is bigger than 90 degree, it is
   set to its supplementary angle.
3. Revised ProjAlign/GReproj.cu. In forward projection, when the ray
   has less than 80% length through the volume, the projection is
   marked with negative large number.
4. PatchAlign/CFitPatchShifts.cpp: Remove fitting tilt axis. It can
   cause large error when there are enough and well spread patches.

Version 1.0.7
-------------
1. Changed ProjAlign/GReproj.cu: 
   1) fW is the square of cosine of the angular difference in mGBackProj
   2) all the aligned images are weighted and then involved in mGBackProj
   3) When ang. difference is larger than 90 deg., its supplemental
      angle is used for the calculation of weights.
2. CInput.cpp: AlignZ is set 400 by default instead of 800.

Version 1.0.8
-------------
1. CInput.cpp: AlignZ set to 600.
2. MrcUtil/CLocalAlignParam::GetParam: Added screen for abnormal large shift.
   for any shift with the magnitude larger than 100 pixel, it will not be
   included in the local motion correction in Correct/GCorrPatchShift.cu.
3. MrcUtil/CRemoveDarkFrames.cpp: Detect dark tilt images based upon the
   ratio of image mean and sigma. Using the ratio at zero-tilt image as the
   reference.
4. ProjAlign/GReproj.cu: Changed back to power of 4 and excluding projections
   whose angular difference is bigger than 12 degree.
5. Correct/GCorrPatchShift.cu::mGRandom: Reduced number of loops to 20. MUCH
   FASTER now!

Version 1.0.9
-------------
1. Need to add screening for bad shifts when fitting local motion to models.
   This has been implemented in PatchAlign/CFitPatchShifts.cpp

Version 1.0.10
--------------
1. ProjAlign/CProjAlignMain.cpp: Allow users to set correlation image size
   in ProjAlign/CParam::m_fXcfSize.
2. In PatchAlign/CLocalAlign.cpp: The first round of alignment uses 1024
   for Xcf size. The second round uses 2048.

Version 1.0.11
--------------
1. Bug fix: Correct/GCorrPatchShift.cu::mGRandom: ix > iInImgX and 
   iy > giInSize[1] are replaced with " >= ".

Version 1.0.12
--------------
1. Adopted IMOD handedness. Prior to this version, the z axis is flipped
   180 degree with respect to IMOD.
   Files changed: Recon/CDoSartRecon.cpp, CDoWbpRecon.cpp
2. Output volume/tilt series with pixel size stored.
3. -TiltAxis takes two values: tiltAxis refine.
   refine = -1: no refinement at all
   refine = 0:  single tilt axis throughout the tilt series
   refine = 1:  variable tilt axis throughout the tilt series.

Version 1.1.0
-------------
1. Added "-OutImod" function that output required filed needed for Relion4.
2. Tilt axis determination: when user specifies an initial value, AreTomo
   will not add 180 to the refined value even when it is less than -45 degree.
3. Fixed the bug in CInput.cpp when parsing the input of tilt axis. When a
   single value is entered, it was actually not parsed. 
4. Added "-DarkTol" tag allowing users to change the threshold in detecting
   and removing dark images.
5. Fixed the bug that does not generate tilt file when the -AlnFile is
   present in command line.
6. Removed padding in x axis of the volume to be reconstructed in Recon/
   CDoSartRecon.cpp and CDoWbpRecon/cpp

Version 1.1.1
-------------
1. Output a text file that lists the dark images removed from the raw
   tilt series.
2. Fixed the bug in MrcUtil/CLoadStack.cpp. When the alignment has less
   images than the input MRC, the extra images are the dark images that
   should not be loaded.

Version 1.1.2
-------------
1. The ".tlt" files generated in CAlignParam.cpp and CImodUtil.cpp are
   inconsisitent when counted based on newlines although there are the
   same number of rows. This can be a problem when the file is counted
   using "wc -l" that counts the newlines. Fixed CImodUtil.cpp.

Version 1.1.3
-------------
1. Implemented automatic detection of high-contrast features for local
   alignment. Added CDetectFeatures.cpp GNormByStd2D.cu in PatchAlign.
2. Revised PatchAlign/CLocalAlign.cpp. Since the features are selected
   at zero degree image after tilt-offset correction, there is no need
   to use m_iRefTilt.

Version 1.2.0
-------------
1. In MrcUtil/CAlignParam.cpp, calculate the geometric rotation around the
   tilt axis due z offset.
2. In ProjAlign, take out the z offset induced motion prior to projection
   matching alignment and add it back to the measurement.
3. Implemented tilt-axis refinement based on local alignment.
4. Temporally took out the function of variable tilt axis across the
   tilt series. Fixed tilt axis only.

Version 1.2.1
-------------
1. In MrcUtil/CAlignParam.cpp, fit the geometric rotation around the tilt axis
   due to x offset (the dominant feature may not be at the prespecified
   location). 
2. In CProcessThread::mAlign, estimate x offset and remove it.
3. In PatchAlign/CLocalAlign, estimate x offset and remove it.

Version 1.2.2
-------------
1. Fixed the bug in ImodUtil. Use the user input of volume hight in the
   generation of tilt.com.
2. Disabled the removal of rotation center x induced offet.

Version 1.2.3
-------------
1. Done multi-GPU implementation for patch alignment. Each GPU is dedicated
   to one patch alignment at a time.

Version 1.2.4
-------------
1. Fixed the bug in flipping the volume enabled by -FlipVol 1. The bug is that
   The flipped x-z view is inverted with respect to the IMod xz view after its
   rotation around x axis.

Version 1.2.5
-------------
1. ProjAlign: applied R-filter in Correct/CCorrTomoStack.cpp. Therefore the
   intermediate reconstruction is R-weighted before it is reprojected as
   the reference in projection matching.

Version 1.3.0
-------------
1. Removed "-OutXF".
2. Expanded "-OutImod" that accepts 1, 2, and 3.
   -OutImod 1: 
      1) specifically for Relion 4 subtomoaveraging. The generated tlt
         file consists of all the tilt angles including those of the 
         removed dark images.
      2) .xf file contains the transformation entries for all the images
         including the dark images that are assigned unit matricies.
      3) tilt.com has a line "EXCLUDELIST" followed 1-based indices of
         dark images.
      4) The order of the contents in .tlt and .xf files corresponds to
         the order of the original tilt series.
   -OutImod 2:
      1) Can be used for both Warp and Relion 4.
      2) A new tilt series consists of the raw images without the dark
         images. The tilt images are ordered according their tilt angles.
      3) Both .tlt and .xf files do not contain any entries for the
         removed dark images.
      4) tilt.com does not have a line for "EXCLUDELIST".
   -OutImod 3:
      1) Specifically for subtomo averaging based upon the global- and
         local-aligned tilt series generated by AreTomo.
      2) A new tilt series consisting of the ALIGNED images without the
         dark images is generated.
      3) .xf file consists only the unit entries for the tilt images.
      4) .tlt file consists only the angles of the images in the
         generated tilt series.
3. A weakened and inversed sinc filter is applied to the raw images to
   enhance the high-frequency components before they are aligned. The
   purpose is to reduce the information loss due to linear interpolation.
4. Constant stretching formulae is used to determined the subset of
   projections is used to reconstruct the intermediate tomogram used
   to calculate the forward projection in ProjAlign::CCalcReproj.cu. 
5. Splitted MrcUtil/CAlignFile.cpp into CSaveAlignFile.cpp and
   CLoadAlignFile.cpp
6. In aln file is added a new column for the local alignment. This
   column specifies if the local alignments are good or bad. The
   bad alignments are excluded in the local motion correction in
   Correct/GCorrPatchShift.cu.
7. In PatchAlign/CFitPatchShifts.cpp, added a screening method that
   checks if there are abnormally large shifts in each patch tilt
   series.
8. In back- and forward-projection, change the coordinate system to be
   compatibable with IMod. The left edge is 0 and the right Nx * 0.5.
   The first pixel coordinate is 0.5 and the last Nx - 1 + 0.5. To
   convert the coordinates to pixel index, we can simply use 
   ix = (int)x, iy = (int)y.
9. When -OutImod is enabled, the tomogram file has been renamed from
   tomogram.mrc to xxx_Vol.mrc where xxx denotes the output MRC named
   following -OutMrc.

Version 1.3.1
-------------
1. Added an option -InptCor that lets users choose whether or not to enable
   the correction of linear interpolation.

Version 1.3.2
-------------
1. Change: MrcUtil/CRemoveDarkFrames.cpp: use the absolute values of means
   to determine the dark frames since tilt images in some tilt series may
   be negative.
1. Change: ProjAlign/CCalcReproj.cpp: reimplemented the determination of the
   projection range.
3. Change: PatchAlign/CFitPatchShifts.cpp: When -TiltAxis xxx -1 is set, do not
   refine tilt axis.
4. Change: Correct/CCorrTomoStack.cpp: add cudaSetDevice() before free GPU
   memorys. The same change is also made in Recon/CDoBaseRecon.cpp.
   Hopefully, this change can fix the smearing tomograms reconstructed on
   A100, A40 cards.

Version 1.3.3
-------------
1. Bug fix: Ampere GPU specific bug. Aligned tilt series or volumes have
   absurdly large values. Normalization of forward Fourier transform is
   not executed in CuUtilFFT::GFFT2D.cu.
   1) Cause: CuUtilFFT was not compiled for compute_80 and compute_86.
   2) CommonLine/CLineSet.cpp: change: cudaFree() is called after the
      same GPU is set.
   3) CommonLine/CLineSet::Setup: It calls cudaSetDevice() on different
      GPUs. Added cudaGetGpu() and cudaSetGpu() at the beginning and end
      to ensure Setup() does not change GPU context.

Version 1.3.4
-------------
1. Added back "-RoiFile" function. Two classes CRoiTargets.cpp and
   CPatchTargets.cpp have been added to PatchAlign folder.

Version 1.3.5
-------------
1. ImodUtil/CSaveTilts.cpp: removed the empty line at the end of Imod 
   tlt file to satisfy Relion 4.
2. ImodUtil/CImodUtil.cpp: change the index of dark image to 1-based
   from 0-based for Relion 4.

Version 1.3.7, Apr 05, 2023
---------------------------
1. ImodUtil/CImodUtil.cpp: swap the image dimensions when the tilt axis
   is more than 45 degree in tilt.com. In Relion4, the image dimensions
   in tilt.com will be the x and y dimensions of the tomogram. (Arthor
   Milo)

Version 1.4.0: Aug 18, 2023
---------------------------
1. Changes: ProjAlign/CCalcReproj.cpp: revised mFindProjRange that excludes
   lower tilt projections in the calculation of the reprojection since they
   are too different from the projection image to be calculated.
   - ProjAlign/GReproj.cu is changed correspondingly.
2. Major change: Integrated GCtfFind into AreTomo. This funcation estimates
   CTF parameters on raw tilt images and saves both spectrum images and
   text files compatible with both CtfFind and Imod.

Version 1.4.1: Aug 28, 2023
---------------------------
Changes:
1. ProjAlign/CCalcReproj.cpp::111: The number of projections is limited to
   10 or less for reprojection in case the angle step is 1 degree
Bug Fix:
1. FindCtf/CCtfTheory.cpp::68: Inside sqrt the amplitude contrast should be
   squared. phase = atan( (Ca / sqrt(1 - Ca * Ca)) ) 
2. FindCtf/GCalcCtf1D.cu and GCalcCtf2D.cu: same mistake as above
3. FindCtf/GCalcCTF2D.cu::112: we should pass in fAddPhase, not fExtPhase.

Version 1.4.2: Sept 13, 2023
----------------------------
Bug Fix:
1.  Correct/GCorrPatchShift.cu:mGCalcLocalShift: fW = expf(-500...). The B
    factor is too large when there are lots of outliers in the local
    measurements. Since the valid measurements are so sparse, a large B
    factor makes one valid measurement overwhelms the others. 
    It is changed to expf(-100...).

Version 1.4.3: Sept 26, 2023
----------------------------
Bug Fix:
1. Generation of Imod folder. Util/CFileName::Setup bug in building m_acFolder
   path.
Changes:
1. FindCtf/CFindDefocus1D and CFindDefocus2D match GCtfFind.

AreTomo2 [Oct 02, 2023]
-----------------------
1. Renamed from AreTomo 1.4.3 to AreTomo2
2. Change: FindCtf/CGenAvgSpectrum.cpp increases the overlapping from 30 to 50.

AreTomo2 1.1.0 [01-28-2024]
---------------------------
1. FindCtf
   1) CFindCtfMain: repeat estimation from scratch if the refinement yields a
      score very close to 0.
   2) CFindCtfBase: low-pass the full spectrum to enhance the Thon rings at
      high resolution. Then truncate into half spectrum for CTF estimation.
   3) In background removal, the box size is reduced to Y / 30 from Y / 15.
2. StreAlign:
   1) CStretchXcf: when measured shift exceeds 25% of the image size, reset
      the shift to zero. A large shift will cause NaN error in tilt offset
      estimation.
3. Removed the dependency of libraries of CuUtil and CuUtilFFT.
4. Added GFFT1D.cu and GFFT2D.cu in Util subfolder
5. Remvoed CCufft2D.cpp. Its functions are provided in GFFT2D.
6. Added LibSrc folder that contains Util and Mrcfile subfolders where the
   source code of libutil.a and libmrcfile.a are provided. Run "make clean"
   followed by "make all" first in Util and then Mrcfile to recompile the
   libxxx.a files.
7. Revised makefile11 and makefile.
8. 02-04-2024:
   1) CFindDefocus1D and CFindDefocus2D: ensure the search within the
      given range.
   2) CFindCtfBase::mRemoveBackground: removed thresholding and reduced
      the B-factor from 10 to 5.
