@echo off
title NeuraForge - Cache Cleanup
color 0A

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║                                                       ║
echo ║           NeuraForge Cache Cleanup Tool              ║
echo ║                                                       ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo This will clear ZLUDA, MIOpen, and Triton cache files.
echo This is recommended when:
echo   - Updating AMD drivers
echo   - Experiencing generation issues
echo   - After installing new ZLUDA version
echo.
echo WARNING: First generation after cleanup will be slower
echo as caches need to be rebuilt.
echo.
pause
echo.

echo Cleaning local cache folders...
if exist "cache\miopen" (
    rmdir /s /q "cache\miopen"
    echo ✓ Cleared cache\miopen
)
if exist "cache\triton" (
    rmdir /s /q "cache\triton"
    echo ✓ Cleared cache\triton
)

echo.
echo Cleaning user cache folders...

set ZLUDA_CACHE=%LOCALAPPDATA%\ZLUDA\ComputeCache
if exist "%ZLUDA_CACHE%" (
    rmdir /s /q "%ZLUDA_CACHE%"
    echo ✓ Cleared ZLUDA ComputeCache
)

set MIOPEN_CACHE=%USERPROFILE%\.miopen
if exist "%MIOPEN_CACHE%" (
    rmdir /s /q "%MIOPEN_CACHE%"
    echo ✓ Cleared MIOpen cache
)

set TRITON_CACHE=%USERPROFILE%\.triton
if exist "%TRITON_CACHE%" (
    rmdir /s /q "%TRITON_CACHE%"
    echo ✓ Cleared Triton cache
)

echo.
echo Recreating cache directories...
mkdir "cache\miopen" 2>nul
mkdir "cache\triton" 2>nul
echo ✓ Cache directories recreated

echo.
echo ══════════════════════════════════════════════════════
echo   Cache cleanup complete!
echo ══════════════════════════════════════════════════════
echo.
echo Caches will be rebuilt on next model run.
echo.
pause
