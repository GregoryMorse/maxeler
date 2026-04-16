#!/bin/sh
#./savebuild.sh PermanentGlynn singleDFE diamond320
mkdir ./$1DFE/builds/bitstream/$3
cp ./$1DFE/$1_$2_XilinxAlveoU250_DFE/_build.log ./$1DFE/builds/bitstream/$3/$2_build.log 
cp ./$1DFE/builds/bitstream/*$2.* ./$1DFE/builds/bitstream/$3
cp ./$1DFE/builds/bitstream/*$2-* ./$1DFE/builds/bitstream/$3
