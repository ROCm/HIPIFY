@echo off
setlocal

for %%i in (FileCheck.exe) do set FILE_CHECK=%%~$PATH:i
if not defined FILE_CHECK (echo      Error: FileCheck.exe not found in PATH. && exit /b 1)

set HIPIFY=%1
set IN_FILE=%2
set TMP_FILE=%3
set CUDA_ROOT=%4
set ROC=%5

set all_args=%*
call set clang_args=%%all_args:*%6=%%
set clang_args=%6%clang_args%

set test_dir=%~dp2
set "test_dir=%test_dir:\=/%"

set compile_commands=compile_commands.json
set json_in=%test_dir%%compile_commands%.in
set json_out=%test_dir%%compile_commands%

if exist %json_in% (
  powershell -Command "(gc %json_in%) -replace '<test dir>', '%test_dir%' -replace '<CUDA dir>', '%CUDA_ROOT%' | Out-File -encoding ASCII %json_out%"
  %HIPIFY% -o=%TMP_FILE% %IN_FILE% %CUDA_ROOT% -p=%test_dir%
) else (
  %HIPIFY% -o=%TMP_FILE% %IN_FILE% %CUDA_ROOT% %ROC% -- %clang_args%
)

if errorlevel 1 (echo      Error: hipify-clang.exe failed with exit code: %errorlevel% && exit /b %errorlevel%)

findstr /v /r /c:"[ ]*//[ ]*[CHECK*|RUN]" %TMP_FILE% | %FILE_CHECK% %IN_FILE%
if errorlevel 1 (echo      Error: FileCheck.exe failed with exit code: %errorlevel% && exit /b %errorlevel%)
