
all: SIM DFE CPU
#SIMZONE DFEZONE CPUZONE

DFE:
	@$(MAKE)  DFE -C ./PermanentGlynnDFE/
	@$(MAKE)  DFEDUAL -C ./PermanentGlynnDFE/

SIM:
	@$(MAKE)  SIM -C ./PermanentGlynnDFE/
	@$(MAKE)  SIMDUAL -C ./PermanentGlynnDFE/

CPU:
	@$(MAKE) MODE=release MAXFILE_DIRS=~/workspace/PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true
	@$(MAKE) MODE=release MAXFILE_DIRS=~/workspace/PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=~/workspace/PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

CPUSIM:
	@$(MAKE) run_sim MAXFILE_DIRS=/home/morse/workspace/PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

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
	@$(MAKE) MODE=release MAXFILE_DIRS=~/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/ SIM=true
	@$(MAKE) MODE=release MAXFILE_DIRS=~/workspace/PermanentZOneDFE/builds/bitstream -C ./PermanentZOneCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=~/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/ SIM=true

CPUSIMZONE:
	@$(MAKE) run_sim MAXFILE_DIRS=~/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/ SIM=true
  
SIMLC:
	@$(MAKE)  SIM -C ./LoopCorrectionDFE/


.PHONY: all

