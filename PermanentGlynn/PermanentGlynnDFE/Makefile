

SIM:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermanentGlynn_singleSIM -Dresult-dir=./builds/simulation -Dengine-params=\"frequency=100 maxDim=48\"

SIMDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermanentGlynn_dualSIM -Dresult-dir=./builds/simulation -Dengine-params=\"dual=true frequency=100 maxDim=48\"

DFE:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermanentGlynn_singleDFE -Dresult-dir=./builds/bitstream -Dengine-params=\"frequency=350 maxDim=48\"

DFEDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermanentGlynn_dualDFE -Dresult-dir=./builds/bitstream -Dengine-params=\"dual=true frequency=350 maxDim=48\"

SIMF:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermanentGlynn_singleSIMF -Dresult-dir=./builds/simulation -Dengine-params=\"useFloat=true frequency=100 maxDim=8\"

SIMFDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermanentGlynn_dualSIMF -Dresult-dir=./builds/simulation -Dengine-params=\"dual=true useFloat=true frequency=100 maxDim=8\"

DFEF:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermanentGlynn_singleDFEF -Dresult-dir=./builds/bitstream -Dengine-params=\"useFloat=true frequency=240 maxDim=40\"

DFEFDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermanentGlynn_dualDFEF -Dresult-dir=./builds/bitstream -Dengine-params=\"dual=true useFloat=true frequency=240 maxDim=40\"

.PHONY:
