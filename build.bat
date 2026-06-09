@echo on
call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Users\bzhao\katago-fork\cpp
echo === CMAKE (OPENCL) ===
C:\Users\bzhao\tools\cmake\cmake-3.30.0-windows-x86_64\bin\cmake.exe -G "NMake Makefiles" ^
  -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCL_INCLUDE_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include" ^
  -DOpenCL_LIBRARY="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64\OpenCL.lib" ^
  -DZLIB_INCLUDE_DIR="C:\Users\bzhao\tools\zlib-1.3.1" ^
  -DZLIB_LIBRARY="C:\Users\bzhao\tools\zlib-1.3.1\zlib.lib" .
echo === CMAKE EXIT: %ERRORLEVEL% ===
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo === BUILD ===
nmake /E
echo === BUILD EXIT: %ERRORLEVEL% ===
