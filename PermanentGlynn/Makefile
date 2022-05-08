
all: SIM DFE CPU
#SIMZONE DFEZONE CPUZONE

DFE:
	@$(MAKE)  DFE -C ./PermanentGlynnDFE/
	@$(MAKE)  DFEDUAL -C ./PermanentGlynnDFE/

DFEF:
	@$(MAKE)  DFEF -C ./PermanentGlynnDFE/
	@$(MAKE)  DFEDUALF -C ./PermanentGlynnDFE/

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
	@$(MAKE)  DFEDUALF -C ./PermRepGlynnDFE/

SIMREP:
	@$(MAKE)  SIM -C ./PermRepGlynnDFE/
	@$(MAKE)  SIMDUAL -C ./PermRepGlynnDFE/

SIMREPF:
	@$(MAKE)  SIMF -C ./PermRepGlynnDFE/
	@$(MAKE)  SIMDUALF -C ./PermRepGlynnDFE/

CPUREP:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true DUAL=true
	#@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/bitstream -C ./PermRepGlynnCPU/
	#@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/bitstream -C ./PermRepGlynnCPU/ DUAL=true
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

.PHONY: all

