@echo off
rem Installation FaceFusion sur le PC Windows (RTX 2080 Ti, CUDA 12)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python install.py --onnxruntime cuda@12 --skip-conda
