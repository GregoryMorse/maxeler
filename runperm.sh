#!/bin/sh
#env LD_LIBRARY_PATH=/home/$USER/workspace/PermanentGlynnCPU/dist/release/lib:$LD_LIBRARY_PATH python3 permanent_benchmark.py
env LD_LIBRARY_PATH=/home/$USER/workspace/PermanentGlynnCPU/dist/release/lib:/home/$USER/workspace/PermRepGlynnCPU/dist/release/lib:$LD_LIBRARY_PATH python3 permanent_benchmark.py

