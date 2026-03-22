@echo off
REM Windows batch script to quickly drop into the WSL tf-gpu environment
REM You can run: `run_in_wsl.bat python src/inference/wsl-layer_image_prediction.py`
wsl bash ./wsl_env.sh %*
