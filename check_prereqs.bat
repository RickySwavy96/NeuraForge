@echo off
title NeuraForge - Prerequisites Checker
color 0B

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║                                                       ║
echo ║        NeuraForge Prerequisites Checker v1.0         ║
echo ║                                                       ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo Checking if your system is ready for NeuraForge...
echo.

set ERRORS=0

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
    echo     ✓ Python found: %PYTHON_VER%
    
    REM Check if it's Python 3.12
    echo %PYTHON_VER% | findstr /C:"3.12" >nul
    if %ERRORLEVEL% EQU 0 (
        echo     ✓ Python 3.12.x detected ^(correct version^)
    ) else (
        echo     ✗ Python version is not 3.12.x ^(recommended: 3.12.9^)
        set /a ERRORS+=1
    )
    
    REM Check if it's from Microsoft Store
    where python | findstr /C:"WindowsApps" >nul
    if %ERRORLEVEL% EQU 0 (
        echo     ✗ WARNING: Python from Microsoft Store detected!
        echo       Please uninstall and use python.org version
        set /a ERRORS+=1
    )
) else (
    echo     ✗ Python not found or not in PATH
    echo       Download from: https://www.python.org/downloads/
    set /a ERRORS+=1
)
echo.

REM Check HIP SDK
echo [2/6] Checking AMD HIP SDK...
if defined HIP_PATH (
    echo     ✓ HIP_PATH found: %HIP_PATH%
    
    if exist "%HIP_PATH%" (
        echo     ✓ HIP SDK directory exists
        
        REM Check for bin folder
        if exist "%HIP_PATH%\bin" (
            echo     ✓ HIP SDK bin folder found
        ) else (
            echo     ✗ HIP SDK bin folder not found
            set /a ERRORS+=1
        )
    ) else (
        echo     ✗ HIP_PATH directory does not exist
        set /a ERRORS+=1
    )
) else (
    echo     ✗ HIP_PATH not set
    echo       Install AMD HIP SDK 6.4 from:
    echo       https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
    set /a ERRORS+=1
)
echo.

REM Check Git
echo [3/6] Checking Git installation...
git --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('git --version') do echo     ✓ %%i
) else (
    echo     ✗ Git not found or not in PATH
    echo       Download from: https://git-scm.com/download/win
    set /a ERRORS+=1
)
echo.

REM Check Visual Studio Build Tools
echo [4/6] Checking Visual Studio Build Tools...
where cl.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo     ✓ Visual Studio Build Tools found
) else (
    echo     ⚠ Visual Studio Build Tools not found in PATH
    echo       Download from: https://aka.ms/vs/17/release/vs_BuildTools.exe
    echo       ^(May still work if installed^)
)
echo.

REM Check AMD GPU
echo [5/6] Checking AMD GPU...
wmic path win32_VideoController get name | findstr /C:"AMD" >nul
if %ERRORLEVEL% EQU 0 (
    for /f "skip=1 tokens=*" %%i in ('wmic path win32_VideoController get name ^| findstr /C:"AMD"') do (
        echo     ✓ AMD GPU detected: %%i
    )
) else (
    echo     ✗ No AMD GPU detected
    echo       NeuraForge requires AMD Radeon GPU
    set /a ERRORS+=1
)
echo.

REM Check disk space
echo [6/6] Checking available disk space...
for /f "tokens=3" %%i in ('dir E:\ ^| findstr /C:"bytes free"') do set FREE_SPACE=%%i
if defined FREE_SPACE (
    echo     ✓ Disk space check complete
    echo       ^(Ensure at least 50GB free for models^)
) else (
    echo     ⚠ Could not check disk space
)
echo.

REM Summary
echo ══════════════════════════════════════════════════════
echo.
if %ERRORS% EQU 0 (
    echo     🎉 All checks passed!
    echo.
    echo     Your system is ready for NeuraForge installation.
    echo.
    echo     Next steps:
    echo     1. Restart your PC if you just installed HIP SDK
    echo     2. Run: python neuraforge_setup.py
    echo     3. Follow the installation prompts
    echo.
) else (
    echo     ⚠ Found %ERRORS% issue^(s^)
    echo.
    echo     Please fix the issues above before installation.
    echo.
    echo     Common fixes:
    echo     - Install missing software
    echo     - Add to PATH environment variable
    echo     - Restart PC after installations
    echo     - Use Python from python.org, not Store
    echo.
)
echo ══════════════════════════════════════════════════════
echo.

REM Additional system info
echo Additional System Information:
echo ═══════════════════════════════
echo.
echo OS Version:
ver
echo.
echo Environment Variables:
echo   HIP_PATH = %HIP_PATH%
echo   HIP_PATH_64 = %HIP_PATH_64%
echo   PATH check = 
echo   %PATH% | findstr /C:"Python" /C:"HIP" /C:"Git"
echo.

pause
