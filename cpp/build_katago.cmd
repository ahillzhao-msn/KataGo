@echo on
@setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
echo VS_ENV_LOADED

set ZLIB_DIR=C:\Users\xiaoj\katago-fork\cpp\external\zlib-src
cd /d C:\Users\xiaoj\katago-fork\cpp

if exist build rmdir /s /q build
mkdir build
cd build

echo START_CMAKE
cmake .. -G "NMake Makefiles" -DUSE_BACKEND=OPENCL -DUSE_AVX2=1 ^
  -DZLIB_INCLUDE_DIR=%ZLIB_DIR% ^
  -DZLIB_LIBRARY=%ZLIB_DIR%\zlib.lib
if %ERRORLEVEL% NEQ 0 echo CMAKE_FAILED & exit /b %ERRORLEVEL%

echo START_NMAKE
nmake
if %ERRORLEVEL% NEQ 0 echo NMAKE_FAILED & exit /b %ERRORLEVEL%

echo BUILD_SUCCEEDED
copy /y katago.exe C:\Users\xiaoj\katago-fork\cpp\katago.exe
copy /y katago.exe C:\Users\xiaoj\.katago\enhanced\katago.exe
echo DEPLOY_DONE
@endlocal
