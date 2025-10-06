#include "ImplicitHeader.h"

#include <sstream>
#include <regex>
#include <set>
#include <vector>
#include <system_error>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

#include "CUDA2HIP.h"       
#include "LLVMCompat.h"     
#include "HipifyAction.h"   

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
  static const std::regex LocalIncludeRe(
      R"(^\s*#\s*include\s*\"([^\"\n]+)\"\s*(?://.*)?$)", std::regex::ECMAScript);

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

static bool runHipifyOnSingleFile(const std::string &srcPath,
                                  const std::string &mainSourceAbsPath,
                                  const ct::CompilationDatabase *compDB,
                                  ct::CommonOptionsParser *OptionsParserPtr,
                                  const char *hipify_exe,
                                  bool overwriteOriginalIfInplace = false) {
  std::error_code EC;

  SmallString<128> tmpFile;
  StringRef srcFileName = sys::path::filename(srcPath);

  if (TemporaryDir.empty()) {
    EC = sys::fs::createTemporaryFile(srcFileName, "hip", tmpFile);
    if (EC) {
      errs() << "\n" << sHipify << sError << "Failed to create temporary file: " << EC.message() << "\n";
      return false;
    }
  } else {
    std::string tmpDirAbs = getAbsoluteDirectoryPath(TemporaryDir, EC);
    if (EC) {
      errs() << "\n" << sHipify << sError << "Temporary dir error: " << EC.message() << "\n";
      return false;
    }
    
    SmallString<256> tmpTemplate(tmpDirAbs);
    tmpTemplate.push_back('/');
    tmpTemplate.append(srcFileName);
    tmpTemplate.append(".XXXXXX.hip");
    EC = sys::fs::createUniqueFile(tmpTemplate, tmpFile);
    if (EC) {
      EC = sys::fs::createTemporaryFile(srcFileName, "hip", tmpFile);
      if (EC) {
        errs() << "\n" << sHipify << sError << "Failed to create temporary file: " << EC.message() << "\n";
        return false;
      }
    }
  }

  EC = sys::fs::copy_file(srcPath, tmpFile);
  if (EC) {
    errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << srcPath << " to " << tmpFile << "\n";
    if (!SaveTemps) sys::fs::remove(tmpFile);
    return false;
  }

  // Run RefactoringTool on temp file
  ct::RefactoringTool Tool((compDB ? *compDB : OptionsParserPtr->getCompilations()), std::string(tmpFile.c_str()));
  ct::Replacements &replacementsToUse = llcompat::getReplacements(Tool, tmpFile.c_str());
  ReplacementsFrontendActionFactory<HipifyAction> actionFactory(&replacementsToUse);

  if (!appendArgumentsAdjusters(Tool, mainSourceAbsPath, hipify_exe)) {
    errs() << "\n" << sHipify << sError << "LLVM/resource config failed for header: " << srcPath << "\n";
    if (!SaveTemps) sys::fs::remove(tmpFile);
    return false;
  }

  if (Tool.runAndSave(&actionFactory)) {
    errs() << "\n" << sHipify << sError << "Hipifying header failed: " << srcPath << "\n";
    if (!SaveTemps) sys::fs::remove(tmpFile);
    return false;
  }

  SmallString<256> dstHipPath(srcPath);
  dstHipPath += ".hip";
  std::string dstHip = std::string(dstHipPath.str());

  EC = sys::fs::copy_file(tmpFile, dstHip);
  if (EC) {
    errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFile << " to " << dstHip << "\n";
    if (!SaveTemps) sys::fs::remove(tmpFile);
    return false;
  }

  // Only overwrite original if explicitly requested (and Inplace is set)
  if (Inplace && overwriteOriginalIfInplace) {
    EC = sys::fs::copy_file(tmpFile, srcPath);
    if (EC) {
      errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFile << " to " << srcPath << "\n";
      if (!SaveTemps) sys::fs::remove(tmpFile);
      return false;
    }
  }

  if (!SaveTemps) {
    sys::fs::remove(tmpFile);
  }

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

  std::vector<std::string> work(initial.begin(), initial.end());
  std::set<std::string> processed;

  while (!work.empty()) {
    std::string hdr = work.back();
    work.pop_back();
    if (processed.count(hdr)) {
      errs() << sHipify << sWarning << "Duplicate local header reference ignored: " << hdr << "\n";
      continue;
    }
    processed.insert(hdr);

    std::string original;
    if (!readFile(hdr, original)) {
      errs() << sHipify << sError << "Cannot read header: " << hdr << "\n";
      continue;
    }

    bool ok = runHipifyOnSingleFile(hdr, mainSourceAbsPath, compDB,
                                    OptionsParserPtr, hipify_exe, false);
    if (!ok) {
      errs() << sHipify << sError << "Hipify failed for header: " << hdr << "\n";
      return false;
    }

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

  return true;
}
