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

#include "HeaderInjection.h"
#include "LocalHeader.h"
#include "LLVMCompat.h"

#include <sstream>
#include <regex>
#include <fstream>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace std;

namespace {

// Matches system/library includes
static const std::regex SystemIncludeRe(
    R"(^\s*#\s*include\s*<([^>\n]+)>)", std::regex::ECMAScript);

// Matches local (quoted) includes
static const std::regex LocalIncludeRe(
    R"(^\s*#\s*include\s*\"([^\"\n]+)\")", std::regex::ECMAScript);

// Matches #ifndef guard
static const std::regex IfndefGuardRe(
    R"(^\s*#\s*ifndef\s+(\w+)\s*$)", std::regex::ECMAScript);

// Matches #define for guard
static const std::regex DefineGuardRe(
    R"(^\s*#\s*define\s+(\w+)\s*$)", std::regex::ECMAScript);

static const std::regex PragmaOnceRe(
    R"(^\s*#\s*pragma\s+once\s*$)", std::regex::ECMAScript);

bool readFileContent(const std::string &path, std::string &out) {
  auto MBOrErr = llvm::MemoryBuffer::getFile(path);
  if (!MBOrErr) return false;
  out = MBOrErr->get()->getBuffer().str();
  return true;
}

std::string extractIncludePath(const std::string &line) {
  std::smatch m;
  if (std::regex_search(line, m, SystemIncludeRe)) {
    return m[1].str();
  }
  return "";
}

}

bool collectPrecedingSystemIncludes(const std::string &mainSourceAbsPath,
                                     const std::string &targetHeaderAbsPath,
                                     std::vector<std::string> &outSystemIncludes) {
  std::string content;
  if (!readFileContent(mainSourceAbsPath, content)) {
    errs() << sHipify << sError << "Cannot read source file: " << mainSourceAbsPath << "\n";
    return false;
  }

  std::string targetFileName = std::string(sys::path::filename(targetHeaderAbsPath));

  std::istringstream iss(content);
  std::string line;
  std::smatch sysMatch, localMatch;

  while (std::getline(iss, line)) {
    if (std::regex_search(line, localMatch, LocalIncludeRe)) {
      std::string localInc = localMatch[1].str();
      std::string localFileName = std::string(sys::path::filename(localInc));
      if (localFileName == targetFileName) {
        break;
      }
      continue;
    }

    if (std::regex_search(line, sysMatch, SystemIncludeRe)) {
      outSystemIncludes.push_back(line);
    }
  }

  return true;
}

void detectIncludeGuard(const std::string &headerContent,
                        size_t &guardEndLine,
                        std::string &guardType) {
  guardEndLine = 0;
  guardType = "none";

  std::istringstream iss(headerContent);
  std::string line;
  size_t lineNum = 0;
  std::string ifndefSymbol;

  while (std::getline(iss, line)) {
    std::smatch m;

    if (std::regex_match(line, PragmaOnceRe)) {
      guardType = "pragma_once";
      guardEndLine = lineNum;
      return;
    }

    if (std::regex_match(line, m, IfndefGuardRe)) {
      ifndefSymbol = m[1].str();
      for (int i = 0; i < 5 && std::getline(iss, line); ++i) {
        lineNum++;
        if (std::regex_match(line, m, DefineGuardRe)) {
          if (m[1].str() == ifndefSymbol) {
            guardType = "ifndef";
            guardEndLine = lineNum;
            return;
          }
        }
        if (line.empty() || line.find("//") == 0 || line.find("/*") == 0) {
          continue;
        }
        break;
      }
    }

    lineNum++;
  }
}

void getExistingIncludes(const std::string &headerContent,
                         std::set<std::string> &existingIncludes) {
  std::istringstream iss(headerContent);
  std::string line;
  std::smatch m;

  while (std::getline(iss, line)) {
    if (std::regex_search(line, m, SystemIncludeRe)) {
      existingIncludes.insert(m[1].str());
    }
  }
}

bool createInjectedHeader(const std::string &mainSourceAbsPath,
                          const std::string &targetHeaderAbsPath,
                          const std::string &injectedFilePath) {
  std::string headerContent;
  if (!readFileContent(targetHeaderAbsPath, headerContent)) {
    errs() << sHipify << sError << "Cannot read target header: " << targetHeaderAbsPath << "\n";
    return false;
  }

  std::vector<std::string> systemIncludes;
  if (!collectPrecedingSystemIncludes(mainSourceAbsPath, targetHeaderAbsPath,
                                       systemIncludes)) {
  }

  std::set<std::string> existingIncludes;
  getExistingIncludes(headerContent, existingIncludes);

  std::vector<std::string> uniqueIncludes;
  for (const auto &inc : systemIncludes) {
    std::string path = extractIncludePath(inc);
    if (!path.empty() && existingIncludes.find(path) == existingIncludes.end()) {
      uniqueIncludes.push_back(inc);
      existingIncludes.insert(path);
    }
  }

  if (uniqueIncludes.empty()) {
    std::ofstream out(injectedFilePath);
    if (!out.is_open()) {
      errs() << sHipify << sError << "Cannot create injected file: " << injectedFilePath << "\n";
      return false;
    }
    out << headerContent;
    out.close();
    return true;
  }

  size_t guardEndLine;
  std::string guardType;
  detectIncludeGuard(headerContent, guardEndLine, guardType);

  std::string mainFileName = std::string(sys::path::filename(mainSourceAbsPath));
  std::ostringstream injection;
  injection << "// --- HIPIFY: Injected dependencies from " << mainFileName << " ---\n";
  for (const auto &inc : uniqueIncludes) {
    injection << inc << "\n";
  }
  injection << "// --- End injected dependencies ---\n";
  injection << "\n";

  std::ofstream out(injectedFilePath);
  if (!out.is_open()) {
    errs() << sHipify << sError << "Cannot create injected file: " << injectedFilePath << "\n";
    return false;
  }

  std::istringstream iss(headerContent);
  std::string line;
  size_t lineNum = 0;
  bool injected = false;

  while (std::getline(iss, line)) {
    out << line << "\n";

    if (!injected && lineNum == guardEndLine && guardType != "none") {
      out << injection.str();
      injected = true;
    }

    lineNum++;
  }

  if (!injected && guardType == "none") {
    out.close();
    std::ofstream outNew(injectedFilePath);
    if (!outNew.is_open()) {
      errs() << sHipify << sError << "Cannot create injected file: " << injectedFilePath << "\n";
      return false;
    }
    outNew << injection.str();
    outNew << headerContent;
    outNew.close();
  } else {
    out.close();
  }

  return true;
}

bool hipifyHeaderWithInjection(const std::string &headerAbsPath,
                               const std::string &outputPath,
                               const std::string &mainSourceAbsPath,
                               const ct::CompilationDatabase *compDB,
                               ct::CommonOptionsParser *OptionsParserPtr,
                               const char *hipify_exe) {
  std::string headerStem = std::string(sys::path::stem(headerAbsPath));
  std::string headerExt = std::string(sys::path::extension(headerAbsPath));
  
  if (!headerExt.empty() && headerExt[0] == '.') {
    headerExt = headerExt.substr(1);
  }
  if (headerExt.empty()) {
    headerExt = "h";
  }
  
  std::string tempPrefix = "inject_" + headerStem;
  
  SmallString<256> injectedPath;
  std::error_code EC = sys::fs::createTemporaryFile(tempPrefix, headerExt, injectedPath);
  if (EC) {
    errs() << sHipify << sError << "Cannot create temporary file: " << EC.message() << "\n";
    return false;
  }

  if (!createInjectedHeader(mainSourceAbsPath, headerAbsPath, std::string(injectedPath.str()))) {
    sys::fs::remove(injectedPath);
    return false;
  }

  bool hipifyOk = hipifySingleSource(
      std::string(injectedPath.str()),
      outputPath,
      compDB,
      OptionsParserPtr,
      hipify_exe,
      mainSourceAbsPath,
      false
  );

  sys::fs::remove(injectedPath);

  if (!hipifyOk) {
    errs() << sHipify << sError << "Failed to hipify (injection): " << headerAbsPath << "\n";
    return false;
  }

  return true;
}

