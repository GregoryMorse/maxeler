SIM:
	ant build -Dmanager=permanenttest.PermanentTestManager -Dtarget=DFE_SIM -Dmaxfile-name=PermanentTest_singleSIM -Dresult-dir=./builds/simulation -Dengine-params=\"frequency=100 inpX=128 inpY=128 useFloat=false isComplex=false addSubMul=2 strategy=1\"

DFE:
	ant build -Dmanager=permanenttest.PermanentTestManager -Dtarget=DFE -Dmaxfile-name=PermanentTest_singleDFE -Dresult-dir=./builds/bitstream -Dengine-params=\"frequency=500 inpX=66 inpY=66 useFloat=false isComplex=false addSubMul=3 strategy=1\"
