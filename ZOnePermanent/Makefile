
all: SIMZONE CPUZONE

SIMZONE:
	@$(MAKE)  SIM -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWS -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWSGRAY -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMGLYNN -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMGLYNNGRAY -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMDUAL -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMGRAYDUAL -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMROWSDUAL -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMROWSGRAYDUAL -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMGLYNNDUAL -C ./PermanentZOneDFE/
	#@$(MAKE)  SIMGLYNNGRAYDUAL -C ./PermanentZOneDFE/

CPUZONE:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/morse/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=/home/morse/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/
  
.PHONY: all

