
all: SIMZONE DFEZONE CPUZONE

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
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/morse/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/morse/workspace/PermanentZOneDFE/builds/bitstream -C ./PermanentZOneCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=/home/morse/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/
  
.PHONY: all

