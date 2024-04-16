@echo off
setlocal

for %%i in (FileCheck.exe) do set FILE_CHECK=%%~$PATH:i
if not defined FILE_CHECK (echo      Error: FileCheck.exe not found in PATH. && exit /b 1)

set HIPIFY=%1
set IN_FILE=%2
set TMP_FILE=%3
set CUDA_ROOT=%4
set CLANG_RES=%5
set NUM=%6
set all_args=%*

if %NUM% EQU 1 (
  set HIPIFY_OPTS=%7
  call set clang_args=%%all_args:*%7=%%
) else if %NUM% EQU 2 (
  set HIPIFY_OPTS=%7 %8
  call set clang_args=%%all_args:*%8=%%
) else if %NUM% EQU 3 (
  set HIPIFY_OPTS=%7 %8 %9
  shift
  call set clang_args=%%all_args:*%9=%%
) else if %NUM% EQU 4 (
  shift
  call set HIPIFY_OPTS=%%6 %%7 %%8 %%9
) else if %NUM% EQU 5 (
  shift
  shift
  call set HIPIFY_OPTS=%%5 %%6 %%7 %%8 %%9
) else (
  set clang_args=%%all_args:*%6=%%
  set NUM=0
)
if %NUM% EQU 4 (
  shift
  call set clang_args=%%all_args:*%9=%%
)
if %NUM% EQU 5 (
  shift
  call set clang_args=%%all_args:*%9=%%
)

set test_dir=%~dp2
set "test_dir=%test_dir:\=/%"

set compile_commands=compile_commands.json
set json_in=%test_dir%%compile_commands%.in
set json_out=%test_dir%%compile_commands%

if exist %json_in% (
  powershell -Command "(gc %json_in%) -replace '<test dir>', '%test_dir%' -replace '<CUDA dir>', '%CUDA_ROOT%' | Out-File -encoding ASCII %json_out%"
  set hipify_cmd=%HIPIFY% -o=%TMP_FILE% %IN_FILE% %CUDA_ROOT% --clang-resource-directory=%CLANG_RES% %HIPIFY_OPTS% -p=%test_dir% %HIPIFY_OPTS% -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1
) else (
  set hipify_cmd=%HIPIFY% -o=%TMP_FILE% %IN_FILE% %CUDA_ROOT% --clang-resource-directory=%CLANG_RES% %HIPIFY_OPTS% -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1 -- %clang_args%
)

SET CUDA=%CUDA_ROOT:~13,-1%

echo [HIPIFY] CUDA directory      : %CUDA%
echo [HIPIFY] clang res directory : %CLANG_RES%
echo [HIPIFY] hipify options count: %NUM%
echo [HIPIFY] hipify options      : %HIPIFY_OPTS%
echo [HIPIFY] clang  options      : %clang_args%
echo [HIPIFY] hipify-clang command: %hipify_cmd%

call %hipify_cmd%

if errorlevel 1 (echo      Error: hipify-clang.exe failed with exit code: %errorlevel% && exit /b %errorlevel%)

findstr /v /r /c:"[ ]*//[ ]*[CHECK*|RUN]" %TMP_FILE% | %FILE_CHECK% %IN_FILE%
if errorlevel 1 (echo      Error: FileCheck.exe failed with exit code: %errorlevel% && exit /b %errorlevel%)
