
all: SIM DFE CPU
#SIMZONE DFEZONE CPUZONE

DFE:
	@$(MAKE)  DFE -C ./PermanentGlynnDFE/
	@$(MAKE)  DFEDUAL -C ./PermanentGlynnDFE/

DFEF:
	@$(MAKE)  DFEF -C ./PermanentGlynnDFE/
	@$(MAKE)  DFEFDUAL -C ./PermanentGlynnDFE/

SIM:
	@$(MAKE)  SIM -C ./PermanentGlynnDFE/
	@$(MAKE)  SIMDUAL -C ./PermanentGlynnDFE/

SIMF:
	@$(MAKE)  SIMF -C ./PermanentGlynnDFE/
	@$(MAKE)  SIMFDUAL -C ./PermanentGlynnDFE/

CPU:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true DUAL=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/ DUAL=true
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

CPUF:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true USE_FLOAT=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true DUAL=true USE_FLOAT=true
	#@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/ USE_FLOAT=true
	#@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/ DUAL=true USE_FLOAT=true
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true USE_FLOAT=true

CPUSIM:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

CPUSIMF:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true USE_FLOAT=true

DFEREP:
	@$(MAKE)  DFE -C ./PermRepGlynnDFE/
	@$(MAKE)  DFEDUAL -C ./PermRepGlynnDFE/

DFEREPF:
	@$(MAKE)  DFEF -C ./PermRepGlynnDFE/
	@$(MAKE)  DFEFDUAL -C ./PermRepGlynnDFE/

SIMREP:
	@$(MAKE)  SIM -C ./PermRepGlynnDFE/
	@$(MAKE)  SIMDUAL -C ./PermRepGlynnDFE/

SIMREPF:
	@$(MAKE)  SIMF -C ./PermRepGlynnDFE/
	@$(MAKE)  SIMFDUAL -C ./PermRepGlynnDFE/

CPUREP:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true DUAL=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/bitstream -C ./PermRepGlynnCPU/
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/bitstream -C ./PermRepGlynnCPU/ DUAL=true
	@$(MAKE) run_sim MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true

CPUREPF:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true USE_FLOAT=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true DUAL=true USE_FLOAT=true
	#@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/bitstream -C ./PermRepGlynnCPU/ USE_FLOAT=true
	#@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/bitstream -C ./PermRepGlynnCPU/ DUAL=true USE_FLOAT=true
	@$(MAKE) run_sim MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true USE_FLOAT=true

CPUSIMREP:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true

CPUSIMREPF:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true USE_FLOAT=true

DFEZONE:
	@$(MAKE)  DFEROWSGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  DFEGLYNNROWSGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  DFEROWSGRAYDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  DFEGLYNNROWSGRAYDUAL -C ./PermanentZOneDFE/

SIMZONE:
	@$(MAKE)  SIM -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWS -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWSGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNN -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNNGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNNROWS -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNNROWSGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  SIMDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGRAYDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWSDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWSGRAYDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNNDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNNGRAYDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNNROWSDUAL -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGLYNNROWSGRAYDUAL -C ./PermanentZOneDFE/

CPUZONE:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/ SIM=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentZOneDFE/builds/bitstream -C ./PermanentZOneCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/ SIM=true

CPUSIMZONE:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/ SIM=true
  
SIMLC:
	@$(MAKE)  SIM -C ./LoopCorrectionDFE/


.PHONY: all

