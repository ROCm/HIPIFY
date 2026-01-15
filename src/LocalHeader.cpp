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

#include "LocalHeader.h"
#include "HeaderInjection.h"
#include "StderrCapture.h"
#include "LLVMCompat.h"

#include <sstream>
#include <regex>
#include <set>
#include <vector>
#include <map>
#include <system_error>
#include <fstream>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace std;
using hipify::StderrCapture;

// Matches local (quoted) includes
static const std::regex LocalIncludeRe(
    R"(^\s*#\s*include\s*\"([^\"\n]+)\"\s*(?://.*)?$)", std::regex::ECMAScript);

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
    if (std::regex_match(line, m, LocalIncludeRe)) {
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
    outs() << sHipify << "No local headers detected in " << mainSourceAbsPath << "\n";
    return true;
  }

  outs() << "\n";
  outs() << sHipify << "Found " << initial.size() << " local header(s) to process\n";
  outs() << sHipify << "Note: Compilation errors during direct attempts may be safely ignored\n";
  outs() << sHipify << "      if the injection fallback succeeds.\n";
  outs() << "\n";
  outs().flush();

  std::vector<std::string> work(initial.begin(), initial.end());
  std::set<std::string> processed;
  
  std::vector<std::string> directSuccess;
  std::vector<std::string> injectionSuccess;
  std::vector<std::string> failed;
  
  // Store captured error output for failed files
  std::map<std::string, std::string> capturedErrors;

  while (!work.empty()) {
    std::string hdr = work.back();
    work.pop_back();
    if (processed.count(hdr)) {
      continue;
    }
    processed.insert(hdr);

    std::string original;
    if (!readFile(hdr, original)) {
      errs() << sHipify << sError << "Cannot read header: " << hdr << "\n";
      failed.push_back(hdr);
      continue;
    }

    std::string hipOut = hdr + ".hip";
    std::string hdrFileName = std::string(sys::path::filename(hdr));
    bool ok = false;

    // HYBRID APPROACH:
    // Step 1: Try direct hipification first (works for self-contained headers)
    // Capture stderr - if both approaches fail, we'll show the errors
    outs() << sHipify << "[" << (directSuccess.size() + injectionSuccess.size() + 1) 
           << "/" << initial.size() << "] Hipifying source: " << hdr << "\n";
    outs().flush();
    
    std::string directErrors;
    {
      // Capture stderr during direct attempt
      StderrCapture capture;
      ok = hipifySingleSource(hdr, hipOut, compDB, OptionsParserPtr,
                               hipify_exe, mainSourceAbsPath, false, true);
      if (!ok) {
        directErrors = capture.getCapturedOutput();
      }
    }

    if (ok) {
      outs() << sHipify << "  -> OK (direct)\n";
      directSuccess.push_back(hdrFileName);
    } else {
      // Step 2: If direct fails, inject preceding includes
      outs() << sHipify << "  -> Trying injection approach...\n";
      outs().flush();
      
      std::string injectionErrors;
      {
        // Capture stderr during injection attempt
        StderrCapture capture;
        ok = hipifyHeaderWithInjection(hdr, hipOut, mainSourceAbsPath,
                                        compDB, OptionsParserPtr, hipify_exe);
        if (!ok) {
          injectionErrors = capture.getCapturedOutput();
        }
      }
      
      if (ok) {
        outs() << sHipify << "  -> OK (injection)\n";
        injectionSuccess.push_back(hdrFileName);
      } else {
        outs() << sHipify << "  -> FAILED\n";
        failed.push_back(hdrFileName);
        
        // Store errors for this file - combine both attempts' errors
        std::string combinedErrors;
        if (!directErrors.empty()) {
          combinedErrors += "=== Direct approach errors ===\n" + directErrors;
        }
        if (!injectionErrors.empty()) {
          if (!combinedErrors.empty()) combinedErrors += "\n";
          combinedErrors += "=== Injection approach errors ===\n" + injectionErrors;
        }
        if (!combinedErrors.empty()) {
          capturedErrors[hdrFileName] = combinedErrors;
        }
      }
    }

    // If recursive, find and queue nested local headers
    if (recursive) {
      std::smatch m;
      std::istringstream iss(original);
      std::string line;
      while (std::getline(iss, line)) {
        if (std::regex_match(line, m, LocalIncludeRe)) {
          std::string rel = m[1].str();
          std::string abs;
          if (resolveLocalIncludeInternal(hdr, rel, abs) &&
              !processed.count(abs))
            work.push_back(abs);
        }
      }
    }
  }

  outs() << "\n";
  outs() << sHipify << "Local Header Hipification Summary\n";
  
  size_t total = directSuccess.size() + injectionSuccess.size() + failed.size();
  size_t success = directSuccess.size() + injectionSuccess.size();
  
  if (!directSuccess.empty()) {
    outs() << sHipify << "  Direct:    " << directSuccess.size() << " header(s)\n";
  }
  if (!injectionSuccess.empty()) {
    outs() << sHipify << "  Injection: " << injectionSuccess.size() << " header(s) (needed deps from main source)\n";
  }
  if (!failed.empty()) {
    outs() << sHipify << "  Failed:  " << failed.size() << " header(s)\n";
    for (const auto &f : failed) {
      outs() << sHipify << "    - " << f << "\n";
    }
  }
  
  outs() << sHipify << "  Total:   " << success << "/" << total << " succeeded\n";
  outs() << "\n";

  // If there were failures, show the captured error details
  if (!failed.empty()) {
    errs() << sHipify << sError << "The following headers failed to hipify:\n";
    errs() << "\n";
    
    for (const auto &f : failed) {
      errs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
      errs() << "  Failed: " << f << "\n";
      errs() << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
      
      auto it = capturedErrors.find(f);
      if (it != capturedErrors.end() && !it->second.empty()) {
        errs() << it->second << "\n";
      } else {
        errs() << "  (No detailed error output captured)\n\n";
      }
    }
    
    errs() << sHipify << "Hint: Check if the headers have:\n";
    errs() << sHipify << "  - Missing #include dependencies\n";
    errs() << sHipify << "  - Syntax errors\n";
    errs() << sHipify << "  - Types/operators not available in HIP\n";
    errs() << "\n";
    
    return false;
  }

  return true;
}
