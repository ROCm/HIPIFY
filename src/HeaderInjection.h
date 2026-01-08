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

#pragma once

#include <string>
#include <vector>
#include <set>
#include "clang/Tooling/CommonOptionsParser.h"

namespace ct = clang::tooling;

bool collectPrecedingSystemIncludes(const std::string &mainSourceAbsPath,
                                     const std::string &targetHeaderAbsPath,
                                     std::vector<std::string> &outSystemIncludes);

void detectIncludeGuard(const std::string &headerContent,
                        size_t &guardEndLine,
                        std::string &guardType);

void getExistingIncludes(const std::string &headerContent,
                         std::set<std::string> &existingIncludes);

bool createInjectedHeader(const std::string &mainSourceAbsPath,
                          const std::string &targetHeaderAbsPath,
                          const std::string &injectedFilePath);

bool hipifyHeaderWithInjection(const std::string &headerAbsPath,
                               const std::string &outputPath,
                               const std::string &mainSourceAbsPath,
                               const ct::CompilationDatabase *compDB,
                               ct::CommonOptionsParser *OptionsParserPtr,
                               const char *hipify_exe);

