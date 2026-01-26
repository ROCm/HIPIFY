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
static const regex SystemIncludeRe(
    R"(^\s*#\s*include\s*<([^>\n]+)>)", regex::ECMAScript);

// Matches local (quoted) includes
static const regex LocalIncludeRe(
    R"(^\s*#\s*include\s*\"([^\"\n]+)\")", regex::ECMAScript);

// Matches #ifndef guard
static const regex IfndefGuardRe(
    R"(^\s*#\s*ifndef\s+(\w+)\s*$)", regex::ECMAScript);

// Matches #define for guard
static const regex DefineGuardRe(
    R"(^\s*#\s*define\s+(\w+)\s*$)", regex::ECMAScript);

static const regex PragmaOnceRe(
    R"(^\s*#\s*pragma\s+once\s*$)", regex::ECMAScript);

bool readFileContent(const string &path, string &out) {
  auto MBOrErr = MemoryBuffer::getFile(path);
  if (!MBOrErr) return false;
  out = MBOrErr->get()->getBuffer().str();
  return true;
}

string extractIncludePath(const string &line) {
  smatch m;
  if (regex_search(line, m, SystemIncludeRe)) {
    return m[1].str();
  }
  return "";
}

}

bool collectPrecedingSystemIncludes(const string &mainSourceAbsPath,
                                     const string &targetHeaderAbsPath,
                                     vector<string> &outSystemIncludes) {
  string content;
  if (!readFileContent(mainSourceAbsPath, content)) {
    errs() << sHipify << sError << "Cannot read source file: " << mainSourceAbsPath << "\n";
    return false;
  }

  string targetFileName = string(sys::path::filename(targetHeaderAbsPath));

  istringstream iss(content);
  string line;
  smatch sysMatch, localMatch;

  while (getline(iss, line)) {
    if (regex_search(line, localMatch, LocalIncludeRe)) {
      string localInc = localMatch[1].str();
      string localFileName = string(sys::path::filename(localInc));
      if (localFileName == targetFileName) {
        break;
      }
      continue;
    }

    if (regex_search(line, sysMatch, SystemIncludeRe)) {
      outSystemIncludes.push_back(line);
    }
  }

  return true;
}

void detectIncludeGuard(const string &headerContent,
                        size_t &guardEndLine,
                        string &guardType) {
  guardEndLine = 0;
  guardType = "none";

  istringstream iss(headerContent);
  string line;
  size_t lineNum = 0;
  string ifndefSymbol;

  while (getline(iss, line)) {
    smatch m;

    if (regex_match(line, PragmaOnceRe)) {
      guardType = "pragma_once";
      guardEndLine = lineNum;
      return;
    }

    if (regex_match(line, m, IfndefGuardRe)) {
      ifndefSymbol = m[1].str();
      for (int i = 0; i < 5 && getline(iss, line); ++i) {
        lineNum++;
        if (regex_match(line, m, DefineGuardRe)) {
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

void getExistingIncludes(const string &headerContent,
                         set<string> &existingIncludes) {
  istringstream iss(headerContent);
  string line;
  smatch m;

  while (getline(iss, line)) {
    if (regex_search(line, m, SystemIncludeRe)) {
      existingIncludes.insert(m[1].str());
    }
  }
}

bool createInjectedHeader(const string &mainSourceAbsPath,
                          const string &targetHeaderAbsPath,
                          const string &injectedFilePath) {
  string headerContent;
  if (!readFileContent(targetHeaderAbsPath, headerContent)) {
    errs() << sHipify << sError << "Cannot read target header: " << targetHeaderAbsPath << "\n";
    return false;
  }

  vector<string> systemIncludes;
  if (!collectPrecedingSystemIncludes(mainSourceAbsPath, targetHeaderAbsPath,
                                       systemIncludes)) {
  }

  set<string> existingIncludes;
  getExistingIncludes(headerContent, existingIncludes);

  vector<string> uniqueIncludes;
  for (const auto &inc : systemIncludes) {
    string path = extractIncludePath(inc);
    if (!path.empty() && existingIncludes.find(path) == existingIncludes.end()) {
      uniqueIncludes.push_back(inc);
      existingIncludes.insert(path);
    }
  }

  if (uniqueIncludes.empty()) {
    ofstream out(injectedFilePath);
    if (!out.is_open()) {
      errs() << sHipify << sError << "Cannot create injected file: " << injectedFilePath << "\n";
      return false;
    }
    out << headerContent;
    out.close();
    return true;
  }

  size_t guardEndLine;
  string guardType;
  detectIncludeGuard(headerContent, guardEndLine, guardType);

  string mainFileName = string(sys::path::filename(mainSourceAbsPath));
  ostringstream injection;
  injection << "// --- HIPIFY: Injected dependencies from " << mainFileName << " ---\n";
  for (const auto &inc : uniqueIncludes) {
    injection << inc << "\n";
  }
  injection << "// --- End injected dependencies ---\n";
  injection << "\n";

  ofstream out(injectedFilePath);
  if (!out.is_open()) {
    errs() << sHipify << sError << "Cannot create injected file: " << injectedFilePath << "\n";
    return false;
  }

  istringstream iss(headerContent);
  string line;
  size_t lineNum = 0;
  bool injected = false;

  while (getline(iss, line)) {
    out << line << "\n";

    if (!injected && lineNum == guardEndLine && guardType != "none") {
      out << injection.str();
      injected = true;
    }

    lineNum++;
  }

  if (!injected && guardType == "none") {
    out.close();
    ofstream outNew(injectedFilePath);
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

bool hipifyHeaderWithInjection(const string &headerAbsPath,
                               const string &outputPath,
                               const string &mainSourceAbsPath,
                               const ct::CompilationDatabase *compDB,
                               ct::CommonOptionsParser *OptionsParserPtr,
                               const char *hipify_exe) {
  string headerStem = string(sys::path::stem(headerAbsPath));
  string headerExt = string(sys::path::extension(headerAbsPath));
  
  if (!headerExt.empty() && headerExt[0] == '.') {
    headerExt = headerExt.substr(1);
  }
  if (headerExt.empty()) {
    headerExt = "h";
  }
  
  string tempPrefix = "inject_" + headerStem;
  
  SmallString<256> injectedPath;
  error_code EC = sys::fs::createTemporaryFile(tempPrefix, headerExt, injectedPath);
  if (EC) {
    errs() << sHipify << sError << "Cannot create temporary file: " << EC.message() << "\n";
    return false;
  }

  if (!createInjectedHeader(mainSourceAbsPath, headerAbsPath, string(injectedPath.str()))) {
    sys::fs::remove(injectedPath);
    return false;
  }

  bool hipifyOk = hipifySingleSource(
      string(injectedPath.str()),
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

