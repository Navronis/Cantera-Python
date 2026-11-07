#!/usr/bin/env bash
set -e
cd /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python

g++ -O2 -std=c++20 tools/run_native_kinetics_oracle.cpp \
    -I/home/nebula/cantera-pinned-oracle/include \
    -L/home/nebula/cantera-pinned-oracle/build/lib \
    -lcantera_shared \
    -Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib \
    -o tools/run_native_kinetics_oracle

echo "Compilation successful: tools/run_native_kinetics_oracle"
