#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <set>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "CUDA2HIP.h"
#include "CUDA2HIP_Scripting.h"
#include "LLVMCompat.h"
#include "HipifyAction.h"
#include "ArgParse.h"
#include "StringUtils.h"
#include "llvm/Support/Debug.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

namespace ct = clang::tooling;

extern bool appendArgumentsAdjusters(ct::RefactoringTool &Tool, const std::string &sSourceAbsPath, const char *hipify_exe);

bool hipifyLocalHeaders(const std::string &srcPath,
                        const ct::CompilationDatabase *compDB,
                        ct::CommonOptionsParser *OptionsParserPtr,
                        const char *hipify_exe,
                        bool recursive = false);

bool resolveLocalInclude(const std::string &mainSourceAbsPath,
                         const std::string &includeToken,
                         std::string &outAbsPath);

bool collectLocalQuotedIncludes(const std::string &mainSourceAbsPath,
                                std::vector<std::string> &outHeaders);