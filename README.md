# AreTomo2
AreTomo2, a multi-GPU accelerated software package that fully automates motion-corrected marker-free tomographic alignment and reconstruction, now includes robust GPU-accelerated CTF (Contrast Transfer Function) estimation in a single package. AreTomo2  is part of our endeavor to build a fully-automated high-throughput processing pipeline that enables real-time reconstruction of tomograms in parallel with tomographic data collection. It strives to be fast and accurate, as well as provides for easy integration into subtomogram processing workflows by generating IMod compatible files containing alignment and CTF parameters needed to bootstrap subtomogram averaging. AreTomo2 can also be used for on-the-fly reconstruction of tomograms and CTF estimation in parallel with tilt series collection, enabling real-time assessment of sample quality and adjustment of collection parameters.

![ReadmeImg](https://github.com/czimaginginstitute/AreTomo2/blob/main/docs/ReadmeImg.png)

An example of AreTomo2 reconstructed tomogram. For more details, please refer to “AreTomo: An integrated software package for automated marker-free, motion-corrected cryo-electron tomographic alignment and reconstruction”, J. Struct Biology:  X Vol 6, 2022

## Installation
AreTomo2 is developed on Linux platform equipped with at least one Nvidia GPU card. To compile from the source, follow the steps below:

1.	git clone https://github.com/czimaginginstitute/AreTomo2.git
2.	cd AreTomo2 
3.	make exe -f makefile11 [CUDAHOME=path/cuda-xx.x]

If the compute capability of GPUs is 5.x, use makefile instead. If CUDAHOME is not provided, the default installation path of CUDA given in makefile or makefile11 will be used.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
