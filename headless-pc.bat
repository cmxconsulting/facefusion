@echo off
rem Encodage sans interface : headless-pc.bat photo.jpg video.mp4 sortie.mp4
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if "%~3"=="" (
	echo Usage : headless-pc.bat source.jpg cible.mp4 sortie.mp4
	exit /b 1
)
python facefusion.py headless-run --config-path facefusion-pc.ini ^
	-s "%~1" -t "%~2" -o "%~3" ^
	--temp-path "%USERPROFILE%\Downloads\tmp\tmp"
