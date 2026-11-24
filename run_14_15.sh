#!/bin/bash
# usage: bash run_14_15.sh   (run inside "paul project" with the venv active)
python3 14_shift_sweep.py ptbxl_out chapman_out 5 40 > shift.log 2>&1
python3 15_second_arch.py ptbxl_out chapman_out 5 40 > arch.log 2>&1
