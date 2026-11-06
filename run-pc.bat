@echo off
rem Interface web FaceFusion sur le PC (profil facefusion-pc.ini)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if not exist "%USERPROFILE%\Downloads\tmp\2" mkdir "%USERPROFILE%\Downloads\tmp\2"
if not exist "%USERPROFILE%\Downloads\tmp\tmp" mkdir "%USERPROFILE%\Downloads\tmp\tmp"
python facefusion.py run --config-path facefusion-pc.ini --open-browser ^
	-o "%USERPROFILE%\Downloads\tmp\2" ^
	--temp-path "%USERPROFILE%\Downloads\tmp\tmp" ^
	%*
