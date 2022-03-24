

SIM:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermRepGlynn_singleSIM -Dresult-dir=./builds/simulation -Dengine-params=\"\"
  
SIMDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermRepGlynn_dualSIM -Dresult-dir=./builds/simulation -Dengine-params=\"dual=true\"

DFE:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermRepGlynn_singleDFE -Dresult-dir=./builds/bitstream -Dengine-params=\"\"

DFEDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermRepGlynn_dualDFE -Dresult-dir=./builds/bitstream -Dengine-params=\"dual=true\"

.PHONY:
