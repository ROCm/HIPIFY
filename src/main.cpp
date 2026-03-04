/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <fstream>
#include <vector>
#include "CUDA2HIP.h"
#include "CUDA2HIP_Scripting.h"
#include "LLVMCompat.h"
#include "HipifyAction.h"
#include "ArgParse.h"
#include "llvm/Support/Debug.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Compilation.h"
#if LLVM_VERSION_MAJOR >= 22
#include "clang/Driver/CudaInstallationDetector.h"
#endif
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#if LLVM_VERSION_MAJOR >= 16
#include "llvm/TargetParser/Host.h"
#else
#include "llvm/Support/Host.h"
#endif

#include "LocalHeader.h"
#include "ImplicitCudaHeaders.h"

#if LLVM_VERSION_MAJOR < 8
#include "llvm/Support/Path.h"
#endif

#define STRINGIFY(x) #x
#define STRINGIFY_EXPANDED(x) STRINGIFY(x)

constexpr auto DEBUG_TYPE = "cuda2hip";

namespace ct = clang::tooling;

void cleanupHipifyOptions(std::vector<const char*> &args) {
  for (const auto &a : hipifyOptions) {
    args.erase(std::remove(args.begin(), args.end(), "--" + a), args.end());
    args.erase(std::remove(args.begin(), args.end(), "-" + a), args.end());
  }
  for (const auto &a : hipifyOptionsWithTwoArgs) {
    // remove all "-option=value" and "--option=value"
    args.erase(
      std::remove_if(args.begin(), args.end(),
        [a](const std::string &s) { return s.find("--" + a + "=") == 0 || s.find("-" + a + "=") == 0; }
      ),
      args.end()
    );
    // remove all pairs of arguments "--option value" and "-option value"
    auto it = args.erase(
      std::remove_if(args.begin(), args.end(),
        [a](const std::string &s) { return s.find("--" + a) == 0 || s.find("-" + a) == 0; }
      ),
      args.end()
    );
    if (it != args.end()) {
        args.erase(it);
    }
  }
}

void DetectCUDA(const std::unique_ptr<clang::driver::Compilation> &C) {
#if LLVM_VERSION_MAJOR >= 22
  const clang::driver::Driver &driver = C->getDriver();
  clang::driver::CudaInstallationDetector CudaInstallation(driver, llvm::Triple(driver.getTargetTriple()), C->getArgs());
  auto& FS = driver.getVFS();
  if (auto cuda_h_file = FS.getBufferForFile(CudaInstallation.getInstallPath() + "/include/cuda.h"))
    Statistics::setCudaVersion((*cuda_h_file)->getBuffer());
  llvm::errs() << "\n" << sHipify << "CUDA Installation Path: " << CudaInstallation.getInstallPath();
  llvm::errs() << "\n" << sHipify << "CUDA_VERSION: " << Statistics::getCudaVersion() << "\n";
#endif
}

void Init(int argc, const char **argv, std::vector<std::string> &files) {
#if LLVM_VERSION_MAJOR >= 21
  clang::DiagnosticOptions diagOpts;
  clang::TextDiagnosticPrinter diagClient(llvm::errs(), diagOpts);
  clang::DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()), diagOpts, &diagClient, false);
#else
  IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts(new clang::DiagnosticOptions());
  clang::TextDiagnosticPrinter diagClient(llvm::errs(), &*diagOpts);
  clang::DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<clang::DiagnosticIDs>(new clang::DiagnosticIDs()), &*diagOpts, &diagClient, false);
#endif
  std::unique_ptr<clang::driver::Driver> driver(new clang::driver::Driver("", llvm::sys::getDefaultTargetTriple(), Diagnostics));
  std::vector<const char*> Args(argv, argv + argc);
  cleanupHipifyOptions(Args);
  std::unique_ptr<clang::driver::Compilation> C(driver->BuildCompilation(Args));
  DetectCUDA(C);
  if (files.size() < 2) return;
  std::vector<std::string> sortedFiles;
  for (const auto &J : C->getJobs()) {
    if (std::string(J.getCreator().getName()) != "clang") continue;
    const auto &JA = J.getArguments();
    for (size_t i = 0; i < JA.size(); ++i) {
      const auto &A = std::string(JA[i]);
      if (std::find(files.begin(), files.end(), A) != files.end() &&
        i > 0 && std::string(JA[i - 1]) == "-main-file-name") {
        sortedFiles.push_back(A);
      }
    }
  }
  if (sortedFiles.empty()) return;
  std::reverse(sortedFiles.begin(), sortedFiles.end());
  files.assign(sortedFiles.begin(), sortedFiles.end());
}

bool checkLLVM(std::string &path_to_check) {
  const std::string file_name_to_check = "__clang_cuda_runtime_wrapper.h";
  const std::string file_name_to_check_2 = "algorithm";
  const std::string cuda_wrappers_dir = "cuda_wrappers";
  std::string fileToCheck = path_to_check + "/" + file_name_to_check;
  bool bExist = llvm::sys::fs::exists(llvm::Twine(fileToCheck.c_str()));
  if (bExist) {
    fileToCheck = path_to_check + "/" + cuda_wrappers_dir + "/" + file_name_to_check_2;
    bExist = llvm::sys::fs::exists(llvm::Twine(fileToCheck.c_str()));
  }
  return bExist;
}

bool setLLVM(ct::RefactoringTool &Tool, const char *hipify_exe) {
  static int Dummy;
  std::string hipify = llvm::sys::fs::getMainExecutable(hipify_exe, (void*)&Dummy);
  std::string hipify_parent_path = std::string(llvm::sys::path::parent_path(hipify));
  std::string clang_ver = STRINGIFY_EXPANDED(LIB_CLANG_RES);
  std::string clang_res_path, clang_inc_path, fileToCheck;
  const std::string include_dir = "include";
  bool bExist = false;
  // 1. --clang-resource-dir is specified
  if (!ClangResourceDir.empty()) {
    clang_res_path = ClangResourceDir;
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
  // 2. Check for ROCm LLVM
  if (!bExist) {
#if defined(_WIN32)
    // HIP SDK for Windows
    clang_res_path = hipify_parent_path + "/../lib/clang/" + clang_ver;
#else
    // ROCm Linux
    clang_res_path = hipify_parent_path + "/../lib/llvm/lib/clang/" + clang_ver;
#endif
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
#ifndef _WIN32
  if (!bExist) {
    // 2.1. ROCm Linux: hipify-clang standalone package
    clang_res_path = hipify_parent_path + "/../" + include_dir + "/hipify";
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
#endif
  // 3. Check for clang include copied by cmake install
  if (!bExist) {
    clang_res_path = hipify_parent_path;
    clang_inc_path = clang_res_path + "/" + include_dir;
    bExist = checkLLVM(clang_inc_path);
  }
  if (bExist) {
    std::string sRes = "-resource-dir=" + clang_res_path;
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sRes.c_str(), ct::ArgumentInsertPosition::BEGIN));
  }
  return bExist;
}

bool appendArgumentsAdjusters(ct::RefactoringTool &Tool, const std::string &sSourceAbsPath, const char *hipify_exe) {
  if (!setLLVM(Tool, hipify_exe)) {
    llvm::errs() << "\n" << sHipify << sError << "LLVM to work with not found. Hipification is impossible. Exiting. To provide hipify-clang with LLVM to work with, please specify the `--clang-resource-directory` option." << "\n";
    return false;
  }
  if (!IncludeDirs.empty()) {
    for (std::string s : IncludeDirs) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::BEGIN));
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-I", ct::ArgumentInsertPosition::BEGIN));
    }
  }
  if (!MacroNames.empty()) {
    for (std::string s : MacroNames) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::BEGIN));
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-D", ct::ArgumentInsertPosition::BEGIN));
    }
  }
  // Standard c++ to use in hipification by default
  llcompat::setStdCPP(Tool);
  std::string sInclude = "-I" + sys::path::parent_path(sSourceAbsPath).str();
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sInclude.c_str(), ct::ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-fno-delayed-template-parsing", ct::ArgumentInsertPosition::BEGIN));
  if (llcompat::pragma_once_outside_header()) {
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-Wno-pragma-once-outside-header", ct::ArgumentInsertPosition::BEGIN));
  }
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("--cuda-host-only", ct::ArgumentInsertPosition::BEGIN));
  if (!CudaGpuArch.empty()) {
    std::string sCudaGpuArch = "--cuda-gpu-arch=" + CudaGpuArch;
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sCudaGpuArch.c_str(), ct::ArgumentInsertPosition::BEGIN));
  }
  if (!CudaPath.empty()) {
    std::string sCudaPath = "--cuda-path=" + CudaPath;
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sCudaPath.c_str(), ct::ArgumentInsertPosition::BEGIN));
  }

  // Implicit CUDA headers to mimic nvcc behavior
  hipify::addImplicitCudaHeaders(Tool);
  llcompat::addTargetIfNeeded(Tool);
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("cuda", ct::ArgumentInsertPosition::BEGIN));
  Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-x", ct::ArgumentInsertPosition::BEGIN));
  if (Verbose) {
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-v", ct::ArgumentInsertPosition::END));
  }
  Tool.appendArgumentsAdjuster(ct::getClangSyntaxOnlyAdjuster());
  return true;
}

bool hipifySingleSource(const std::string &srcPath,
                               const std::string &dstPath,
                               const ct::CompilationDatabase *compDB,
                               ct::CommonOptionsParser *OptionsParserPtr,
                               const char *hipify_exe_path,
                               const std::string &mainContextPath,
                               bool preserveTemp,
                               const std::vector<std::string> &additionalIncludes) {
  std::error_code EC;
  SmallString<128> tmpFile;
  StringRef srcFileName = sys::path::filename(srcPath);

  EC = sys::fs::createTemporaryFile(srcFileName, "hip", tmpFile);
  if (EC) {
    llvm::errs() << "\n" << sHipify << sError << "Failed to create temporary file: " << EC.message() << "\n";
    return false;
  }

  // Copy source to temp
  EC = sys::fs::copy_file(srcPath, tmpFile);
  if (EC) {
    llvm::errs() << "\n" << sHipify << sError << EC.message() 
                 << ": while copying " << srcPath << " to " << tmpFile << "\n";
    if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
    return false;
  }

  // RefactoringTool operates on the file in-place. Giving it the output path is no good,
  // because that'll break relative includes, and we don't want to overwrite the input file.
  // So what we do is operate on a copy, which we then move to the output.
  ct::RefactoringTool Tool((compDB ? *compDB : OptionsParserPtr->getCompilations()),
                           std::string(tmpFile.c_str()));
  ct::Replacements &replacementsToUse = llcompat::getReplacements(Tool, tmpFile.c_str());
  ReplacementsFrontendActionFactory<HipifyAction> actionFactory(&replacementsToUse);

  if (!appendArgumentsAdjusters(Tool, mainContextPath, hipify_exe_path)) {
    llvm::errs() << "\n" << sHipify << sError 
                 << "LLVM/resource config failed for: " << srcPath << "\n";
    if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
    return false;
  }

  for (auto it = additionalIncludes.rbegin(); it != additionalIncludes.rend(); ++it) {
    Tool.appendArgumentsAdjuster(
        ct::getInsertArgumentAdjuster(it->c_str(),
                                       ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(
        ct::getInsertArgumentAdjuster("-include",
                                       ct::ArgumentInsertPosition::BEGIN));
  }

  // Hipify _all_ the things!
  if (Tool.runAndSave(&actionFactory)) {
    llvm::errs() << "\n" << sHipify << sError 
                 << "Hipifying failed: " << srcPath << "\n";
    if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
    return false;
  }

  // Copy the tmpfile to the output
  if (!dstPath.empty()) {
    EC = sys::fs::copy_file(tmpFile, dstPath);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() 
                   << ": while copying " << tmpFile << " to " << dstPath << "\n";
      if (!SaveTemps && !preserveTemp) sys::fs::remove(tmpFile);
      return false;
    }
  }

  // Remove the tmp file without error chec
  if (!SaveTemps && !preserveTemp) {
    sys::fs::remove(tmpFile);
  }

  return true;
}

bool generatePython() {
  bool bToRoc = TranslateToRoc;
  TranslateToRoc = true;
  bool bToPython = python::generate(GeneratePython);
  TranslateToRoc = bToRoc;
  return bToPython;
}

void printVersions() {
  llvm::errs() << "\n" << sHipify << "Supports ROCm HIP from " << Statistics::getHipVersion(hipVersions::HIP_5000) << " up to " << Statistics::getHipVersion(hipVersions::HIP_LATEST);
  llvm::errs() << "\n" << sHipify << "Supports CUDA Toolkit from " << Statistics::getCudaVersion(cudaVersions::CUDA_70) << " up to " << Statistics::getCudaVersion(cudaVersions::CUDA_LATEST);
  llvm::errs() << "\n" << sHipify << "Supports cuDNN from " << Statistics::getCudaVersion(cudaVersions::CUDNN_705) << " up to " << Statistics::getCudaVersion(cudaVersions::CUDNN_LATEST) << " \n";
}

int main(int argc, const char **argv) {
  std::vector<const char*> new_argv(argv, argv + argc);
  std::string sCompilationDatabaseDir;
  auto it = std::find(new_argv.begin(), new_argv.end(), std::string("-p"));
  bool bCompilationDatabase = it == new_argv.end() ? false : true;
  bool bNoCompilationDatabaseDir = false;
  if (bCompilationDatabase) {
    if (it+1 != new_argv.end()) sCompilationDatabaseDir = *(it+1);
    else bNoCompilationDatabaseDir = true;
  } else {
    for (auto &s : new_argv) {
      std::string str = std::string(s);
      if (str.find("-p=") != std::string::npos) {
        bCompilationDatabase = true;
        sCompilationDatabaseDir = str.substr(3, str.size()-3);
        if (sCompilationDatabaseDir.empty()) {
          bNoCompilationDatabaseDir = true;
        }
        break;
      }
    }
  }
  if (bCompilationDatabase && bNoCompilationDatabaseDir) {
    llvm::errs() << "\n" << sHipify << sError << "Must specify compilation database directory" << "\n";
    return 1;
  }
  if (!bCompilationDatabase && std::find(new_argv.begin(), new_argv.end(), std::string("--")) == new_argv.end()) {
    new_argv.push_back("--");
    new_argv.push_back(nullptr);
    argv = new_argv.data();
    argc++;
  }
  llcompat::PrintStackTraceOnErrorSignal();
#if LLVM_VERSION_MAJOR > 12
  auto cop = ct::CommonOptionsParser::create(argc, argv, ToolTemplateCategory, llvm::cl::ZeroOrMore);
  if (!cop) {
    llvm::errs() << "\n" << sHipify << sError << cop.takeError() << "\n";
    return 1;
  }
  ct::CommonOptionsParser &OptionsParser = cop.get();
#else
  ct::CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::ZeroOrMore);
#endif
  if (!llcompat::CheckCompatibility()) {
    return 1;
  }
  std::unique_ptr<ct::CompilationDatabase> compilationDatabase;
  std::vector<std::string> fileSources;
  if (bCompilationDatabase) {
    std::string serr;
    compilationDatabase = ct::CompilationDatabase::loadFromDirectory(sCompilationDatabaseDir, serr);
    if (nullptr == compilationDatabase.get()) {
      llvm::errs() << "\n" << sHipify << sError << "loading Compilation Database from \"" << sCompilationDatabaseDir << "compile_commands.json\" failed\n";
      return 1;
    }
    fileSources = compilationDatabase->getAllFiles();
  } else {
    fileSources = OptionsParser.getSourcePathList();
  }
  if (fileSources.empty() && !GeneratePerl && !GeneratePython && !GenerateMarkdown && !GenerateCSV && !Versions) {
    llvm::errs() << "\n" << sHipify << sError << "Must specify at least 1 positional argument for source file" << "\n";
    return 1;
  }
  if (Versions) printVersions();
  if (!GenerateMarkdown && !GenerateCSV && !DocFormat.empty()) {
    llvm::errs() << "\n" << sHipify << sError << "Must specify a document type to generate: \"md\" and | or \"csv\"" << "\n";
    return 1;
  }
  if (!perl::generate(GeneratePerl)) {
    llvm::errs() << "\n" << sHipify << sError << "hipify-perl generating failed" << "\n";
    return 1;
  }
  if (!generatePython()) {
    llvm::errs() << "\n" << sHipify << sError << "hipify-python generating failed" << "\n";
    return 1;
  }
  if (!doc::generate(GenerateMarkdown, GenerateCSV)) {
    llvm::errs() << "\n" << sHipify << sError << "Documentation generating failed" << "\n";
    return 1;
  }
  if (fileSources.empty()) {
    return 0;
  }
  std::string dst = OutputFilename, dstDir = OutputDir;
  std::error_code EC;
  std::string sOutputDirAbsPath = getAbsoluteDirectoryPath(OutputDir, EC, "output");
  if (EC) {
    return 1;
  }
  if (!dst.empty()) {
    if (fileSources.size() > 1) {
      llvm::errs() << sHipify << sConflict << "-o and multiple source files are specified\n";
      return 1;
    }
    if (Inplace) {
      llvm::errs() << sHipify << sConflict << "both -o and -inplace options are specified\n";
      return 1;
    }
    if (NoOutput) {
      llvm::errs() << sHipify << sConflict << "both -no-output and -o options are specified\n";
      return 1;
    }
    if (!dstDir.empty()) {
      dst = sOutputDirAbsPath + "/" + dst;
    }
  }
  if (NoOutput && Inplace) {
    llvm::errs() << sHipify << sConflict << "both -no-output and -inplace options are specified\n";
    return 1;
  }
  if (!dstDir.empty() && Inplace) {
    llvm::errs() << sHipify << sConflict << "both -o-dir and -inplace options are specified\n";
    return 1;
  }
  if (Examine) {
    NoOutput = PrintStats = true;
  }
  if (TranslateToRoc)
    TranslateToMIOpen = true;
  int Result = 0;
  SmallString<128> tmpFile;
  StringRef sourceFileName, ext = "hip", csv_ext = "csv";
  std::string sTmpFileName, sSourceAbsPath;
  std::string sTmpDirAbsParh = getAbsoluteDirectoryPath(TemporaryDir, EC);
  if (EC) {
    return 1;
  }
  // Arguments for the Statistics print routines.
  std::unique_ptr<std::ostream> csv = nullptr;
  llvm::raw_ostream *statPrint = nullptr;
  bool create_csv = false;
  if (!OutputStatsFilename.empty()) {
    PrintStatsCSV = true;
    create_csv = true;
  } else {
    if (PrintStatsCSV && fileSources.size() > 1) {
      OutputStatsFilename = "sum_stat.csv";
      create_csv = true;
    }
  }
  if (create_csv) {
    if (!OutputDir.empty()) {
      OutputStatsFilename = sOutputDirAbsPath + "/" + OutputStatsFilename;
    }
    csv = std::unique_ptr<std::ostream>(new std::ofstream(OutputStatsFilename, std::ios_base::trunc));
  }
  if (PrintStats) {
    statPrint = &llvm::errs();
  }
  Init(argc, argv, fileSources);
  for (const auto &src : fileSources) {
    // Create a copy of the file to work on. When we're done, we'll move this onto the
    // output (which may mean overwriting the input, if we're in-place).
    // Should we fail for some reason, we'll just leak this file and not corrupt the input.
    sSourceAbsPath = getAbsoluteFilePath(src, EC);
    if (EC) {
      continue;
    }
    sourceFileName = sys::path::filename(sSourceAbsPath);
    if (dst.empty()) {
      if (Inplace) {
        dst = src;
      } else {
        dst = src + "." + ext.str();
        if (!dstDir.empty()) {
          dst = sOutputDirAbsPath + "/" + sourceFileName.str() + "." + ext.str();
        }
      }
    }
    if (TemporaryDir.empty()) {
      EC = sys::fs::createTemporaryFile(sourceFileName, ext, tmpFile);
      if (EC) {
        llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
        Result = 1;
        continue;
      }
    } else {
      sTmpFileName = sTmpDirAbsParh + "/" + sourceFileName.str() + "." + ext.str();
      tmpFile = sTmpFileName;
    }
    EC = sys::fs::copy_file(src, tmpFile);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << src << " to " << tmpFile << "\n";
      Result = 1;
      continue;
    }
    if (PrintStatsCSV) {
      if (OutputStatsFilename.empty()) {
        OutputStatsFilename = sourceFileName.str() + "." + csv_ext.str();
        if (!OutputDir.empty()) {
          OutputStatsFilename = sOutputDirAbsPath + "/" + OutputStatsFilename;
        }
      }
      if (!csv) {
        csv = std::unique_ptr<std::ostream>(new std::ofstream(OutputStatsFilename, std::ios_base::trunc));
      }
    }
    // Initialise the statistics counters for this file.
    Statistics::setActive(src);
    if (!SkipLocalHeaders) {
      if (!hipifyLocalHeaders(sSourceAbsPath,
                              compilationDatabase.get(),
                              &OptionsParser,
                              argv[0],
                              true)) {
        llvm::errs() << "\n" << sHipify << sError
                     << "Local header hipification failed for: "
                     << sys::path::filename(sSourceAbsPath) << "\n";
        Statistics::current().hasErrors = true;
        Result = 1;
      }
    }
   
    std::string outputPath = NoOutput ? "" : dst;
    if (!hipifySingleSource(src, outputPath,
                            compilationDatabase.get(),
                            &OptionsParser,
                            argv[0],
                            sSourceAbsPath,
                            false, {})) {
      Statistics::current().hasErrors = true;
      Result = 1;
      LLVM_DEBUG(llvm::dbgs() << "Hipification failed for: " << src << "\n");
    }

    Statistics::current().markCompletion();
    Statistics::current().print(csv.get(), statPrint);
    dst.clear();
  }
  if (fileSources.size() > 1) {
    Statistics::printAggregate(csv.get(), statPrint);
  }
  return Result;
}
