#!/bin/bash
# Activates the virtual environment
source ~/tf-gpu/bin/activate

# Navigates to the ML project directory
cd /mnt/y/Projects/Ml_Project

# If arguments are passed, run them (e.g., `python src/inference/...`)
# Otherwise, just start an interactive shell
if [ $# -eq 0 ]; then
    exec bash
else
    "$@"
fi
