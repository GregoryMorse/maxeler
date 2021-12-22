
all: SIMZONE CPUZONE

SIMZONE:
	@$(MAKE)  SIM -C ./PermanentZOneDFE/
	@$(MAKE)  SIMGRAY -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWS -C ./PermanentZOneDFE/
	@$(MAKE)  SIMROWSGRAY -C ./PermanentZOneDFE/

CPUZONE:
	@$(MAKE) MODE=release MAXFILE_DIRS=/home/morse/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=/home/morse/workspace/PermanentZOneDFE/builds/simulation -C ./PermanentZOneCPU/
  
.PHONY: all

