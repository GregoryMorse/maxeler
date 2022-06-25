

SIM:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermRepGlynn_singleSIM -Dresult-dir=./builds/simulation -Dengine-params=\"frequency=280\"
  
SIMDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermRepGlynn_dualSIM -Dresult-dir=./builds/simulation -Dengine-params=\"dual=true frequency=280\"

DFE:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermRepGlynn_singleDFE -Dresult-dir=./builds/bitstream -Dengine-params=\"frequency=280\"

DFEDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermRepGlynn_dualDFE -Dresult-dir=./builds/bitstream -Dengine-params=\"dual=true frequency=280\"
 
SIMF:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermRepGlynn_singleSIMF -Dresult-dir=./builds/simulation -Dengine-params=\"useFloat=true frequency=240\"

SIMFDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE_SIM -Dmaxfile-name=PermRepGlynn_dualSIMF -Dresult-dir=./builds/simulation -Dengine-params=\"dual=true useFloat=true frequency=240\"

DFEF:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermRepGlynn_singleDFEF -Dresult-dir=./builds/bitstream -Dengine-params=\"useFloat=true frequency=240\"

DFEFDUAL:
	ant build -Dmanager=permanentglynn.PermanentGlynnDFEManagerAlveoU250 -Dtarget=DFE -Dmaxfile-name=PermRepGlynn_dualDFEF -Dresult-dir=./builds/bitstream -Dengine-params=\"dual=true useFloat=true frequency=240\"

.PHONY:
