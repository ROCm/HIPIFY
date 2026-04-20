/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "LocalHeader.h"
#include "LLVMCompat.h"

#include <sstream>
#include <regex>
#include <set>
#include <vector>
#include <system_error>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

static std::string normalizeSmallStringPath(SmallString<256> &p) {
  llvm::sys::path::remove_dots(p, true);

  SmallString<256> realBuf;
  std::error_code ec = llvm::sys::fs::real_path(p, realBuf);
  if (!ec) {
    return std::string(realBuf.str());
  }

  return std::string(p.str());
}

static bool pathExists(const std::string &p) {
  SmallString<256> in(p.begin(), p.end());

  SmallString<256> realBuf;
  std::error_code ec = llvm::sys::fs::real_path(in, realBuf);
  if (!ec) return true;

  SmallString<256> norm = in;
  llvm::sys::path::remove_dots(norm, true);
  return llvm::sys::fs::exists(norm);
}

namespace {
  static const std::regex LocalIncludeRe
      (R"re(^\s*#\s*include\s*"([^"\n]+)")re", std::regex::ECMAScript);

  static const std::regex SystemIncludeRe(
      R"(^\s*#\s*include\s*<([^>\n]+)>)", std::regex::ECMAScript);

  bool readFile(const std::string &path, std::string &out) {
    auto MBOrErr = llvm::MemoryBuffer::getFile(path);
    if (!MBOrErr) return false;
    out = MBOrErr->get()->getBuffer().str();
    return true;
  }

  bool resolveLocalIncludeInternal(const std::string &mainSourceAbsPath,
                                  const std::string &includeTok,
                                  std::string &outAbs) {
    SmallString<256> base(mainSourceAbsPath);
    sys::path::remove_filename(base);
    SmallString<256> candidate(base);
    sys::path::append(candidate, includeTok);
    sys::path::remove_dots(candidate, true);
    if (pathExists(std::string(candidate.str()))) {
      outAbs = normalizeSmallStringPath(candidate);
      return true;
    }
    return false;
  }
  
  static bool collectIncludesBefore(const std::string &filePath,
                                    const std::string &stopAtAbsPath,
                                    bool collectLocal,
                                    std::set<std::string> &seen,
                                    std::vector<std::string> &outIncludes) {
    std::string content;
    if (!readFile(filePath, content))
      return false;
    
    std::istringstream iss(content);
    std::string line;
    std::smatch m;

    while (std::getline(iss, line)) {
      if (std::regex_search(line, m, LocalIncludeRe)) {
        std::string abspath;
        if (resolveLocalIncludeInternal(filePath, m[1].str(), abspath)) {
          if (abspath == stopAtAbsPath)
            break;
          if (collectLocal && seen.insert(abspath).second)
            outIncludes.push_back(abspath);
        }
      }
      if (std::regex_search(line, m, SystemIncludeRe)) {
        std::string sysInclude = m[1].str();
        if (seen.insert(sysInclude).second)
          outIncludes.push_back(sysInclude);
      }
    }
    return true;
  }

  bool collectPrecedingIncludes(const std::string &mainSourceAbspath,
                                const std::string &targetHeaderAbspath,
                                std::vector<std::string> &outIncludes) {

    std::set<std::string> seen;

    if (!collectIncludesBefore(mainSourceAbspath, targetHeaderAbspath, true, seen, outIncludes)) {
      errs() << sHipify << sError << "Cannot read source files: "
             << mainSourceAbspath << "\n";
      return false;
    }
    return true;
  }

  void collectAncestorSystemIncludes(
      const std::vector<std::string> &ancestorChain,
      std::vector<std::string> &outIncludes) {

    std::set<std::string> seen(outIncludes.begin(), outIncludes.end());

    for (size_t i = 1; i < ancestorChain.size(); ++i) {
      collectIncludesBefore(ancestorChain[i], ancestorChain[i - 1], false, seen, outIncludes);
    }
  }
}

static std::string
resolveCompileContext(const std::string &parentPath,
                      const std::string &mainSourceAbsPath,
                      const clang::tooling::CompilationDatabase *compDB) {
  if (!compDB)
    return mainSourceAbsPath;

  if (!compDB->getCompileCommands(parentPath).empty())
    return parentPath;

  return mainSourceAbsPath;
}

bool resolveLocalInclude(const std::string &mainSourceAbsPath,
                         const std::string &includeToken,
                         std::string &outAbsPath) {
  return resolveLocalIncludeInternal(mainSourceAbsPath, includeToken, outAbsPath);
}

bool collectLocalQuotedIncludes(const std::string &mainSourceAbsPath,
                                std::vector<std::string> &outHeaders) {
  std::string content;
  if (!readFile(mainSourceAbsPath, content)) {
    errs() << "\n" << sHipify << sError << "Cannot read source file: " << mainSourceAbsPath << "\n";
    return false;
  }

  std::set<std::string> uniq;
  std::smatch m;
  std::istringstream iss(content);
  std::string line;
  while (std::getline(iss, line)) {
    if (std::regex_search(line, m, LocalIncludeRe)) {
      std::string rel = m[1].str();
      std::string abs;
      if (resolveLocalIncludeInternal(mainSourceAbsPath, rel, abs)){
        uniq.insert(abs);
      } else {
        errs() << sHipify << sWarning
               << "Missing local header referenced: \"" << rel
               << "\" in " << mainSourceAbsPath << "\n";
      }
    }
  }
  outHeaders.assign(uniq.begin(), uniq.end());
  return true;
}

bool hipifyLocalHeaders(const std::string &mainSourceAbsPath,
                             const ct::CompilationDatabase *compDB,
                             ct::CommonOptionsParser *OptionsParserPtr,
                             const char *hipify_exe,
                             bool recursive) {

  std::vector<std::string> initial;
  if (!collectLocalQuotedIncludes(mainSourceAbsPath, initial)) {
    return false;
  }
  
  if (initial.empty()) {
    outs() << "\n" << sHipify << "No local headers detected in "
           << sys::path::filename(mainSourceAbsPath) << "\n";
    return true;
  }

  outs() << "\n" << sHipify << "Local headers found: " << initial.size()
         << " in " << sys::path::filename(mainSourceAbsPath) << "\n";
  for (size_t i = 0; i < initial.size(); ++i) {
    outs() << (i + 1) << "/" << initial.size()
           << ": " << sys::path::filename(initial[i]) << "\n";
  }

  std::vector<std::pair<std::string, std::vector<std::string>>> work;
  for (const auto &h : initial) {
    work.push_back({h, std::vector<std::string>{mainSourceAbsPath}});
  }
  std::set<std::string> processed;
  std::set<std::string> queued;
  size_t total = initial.size();
  size_t current = 0;

  for (const auto &h: initial) {
    queued.insert(h);
  }

  while (!work.empty()) {
    std::string hdr = work.back().first;
    std::vector<std::string> ancestorChain = work.back().second;
    work.pop_back();
    std::string parentPath = ancestorChain[0];
    if (processed.count(hdr)) {
      outs() << sHipify << sWarning
             << "Duplicate local header reference ignored: "
             << sys::path::filename(hdr) << "\n";
      continue;
    }
    processed.insert(hdr);
    ++current;

    std::string original;
    if (!readFile(hdr, original)) {
      errs() << "\n" << sHipify << sError
             << "Cannot read header: " << sys::path::filename(hdr) << "\n";
      continue;
    }

    std::string hipOut = hdr + ".hip";
    std::vector<std::string> precedingIncludes;
    collectPrecedingIncludes(parentPath, hdr, precedingIncludes);
    collectAncestorSystemIncludes(ancestorChain, precedingIncludes);
    outs() << "\n" << sHipify << "Hipifying local header [" << current
           << "/" << total << "]: " << sys::path::filename(hdr) << "\n";

    bool ok = hipifySingleSource(
        hdr, hipOut, compDB, OptionsParserPtr, hipify_exe,
        resolveCompileContext(parentPath, mainSourceAbsPath, compDB), false,
        precedingIncludes);

    if (!ok) {
      errs() << "\n" << sHipify << sError
             << "Hipify failed for header [" << current << "/" << total
             << "]: " << sys::path::filename(hdr) << "\n";
      return false;
    }
    outs() << sHipify << "Successfully hipified header file" << "\n";

    if (recursive) {
      std::smatch m;
      std::istringstream iss(original);
      std::string line;
      std::vector<std::string> newHeaders;
      while (std::getline(iss, line)) {
        if (std::regex_search(line, m, LocalIncludeRe)) {
          std::string rel = m[1].str();
          std::string abs;
          if (resolveLocalIncludeInternal(hdr, rel, abs) &&
              !processed.count(abs) && !queued.count(abs)) {
            queued.insert(abs);
            std::vector<std::string> childChain;
            childChain.push_back(hdr);
            childChain.insert(childChain.end(), ancestorChain.begin(), ancestorChain.end());
            work.push_back({abs, childChain});
            newHeaders.push_back(abs);
          }
        }
      }
      if (!newHeaders.empty()) {
        total += newHeaders.size();
        outs() << sHipify << "  Recursive: found " << newHeaders.size()
               << " additional local header(s) in "
               << sys::path::filename(hdr) << "\n";
      }
    }
  }

  outs() << "\n" << sHipify << "Local header hipification complete: "
         << processed.size() << " header(s) processed.\n";
  return true;
}
