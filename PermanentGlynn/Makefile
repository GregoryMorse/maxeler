
all: SIM DFE CPU

DFE:
	@$(MAKE)  DFE -C ./PermanentGlynnDFE/
	@$(MAKE)  DFEDUAL -C ./PermanentGlynnDFE/

SIM:
	@$(MAKE)  SIM -C ./PermanentGlynnDFE/
	@$(MAKE)  SIMDUAL -C ./PermanentGlynnDFE/

CPU:
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true
	@$(MAKE) MODE=release MAXFILE_DIRS=../PermanentGlynnDFE/builds/bitstream -C ./PermanentGlynnCPU/
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

CPUSIM:
	@$(MAKE) run_sim MAXFILE_DIRS=../PermanentGlynnDFE/builds/simulation -C ./PermanentGlynnCPU/ SIM=true

.PHONY: all

