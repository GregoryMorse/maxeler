#!/bin/bash
#hwloc-ps
hwloc-bind --membind node:0 --cpubind node:0 -- $1
