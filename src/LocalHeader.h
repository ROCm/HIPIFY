#pragma once

#include <string>
#include <vector>
#include "clang/Tooling/CommonOptionsParser.h"

namespace ct = clang::tooling;

extern bool hipifySingleSource(const std::string &srcPath,
                               const std::string &dstPath,
                               const ct::CompilationDatabase *compDB,
                               ct::CommonOptionsParser *OptionsParserPtr,
                               const char *hipify_exe_path,
                               const std::string &mainContextPath,
                               bool preserveTemp);

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
