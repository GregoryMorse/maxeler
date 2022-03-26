
all: SIM DFE CPU

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
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

CPUF:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true USE_FLOAT=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/ USE_FLOAT=true
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true USE_FLOAT=true

CPUSIM:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

CPUSIMF:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true USE_FLOAT=true

DFEREP:
	@$(MAKE)  DFE -C ./PermRepGlynnDFE/
	@$(MAKE)  DFEDUAL -C ./PermRepGlynnDFE/

SIMREP:
	@$(MAKE)  SIM -C ./PermRepGlynnDFE/
	@$(MAKE)  SIMDUAL -C ./PermRepGlynnDFE/

CPUREP:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true
	#@$(MAKE) MODE=release MAXFILE_DIRS=../PermRepGlynnDFE/builds/bitstream -C ./PermRepGlynnCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true

CPUSIMREP:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermRepGlynnDFE/builds/simulation -C ./PermRepGlynnCPU/ SIM=true

.PHONY: all

