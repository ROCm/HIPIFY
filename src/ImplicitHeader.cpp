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

static bool readFileToString(const std::string &path, std::string &out) {
  auto MBOrErr = llvm::MemoryBuffer::getFile(path);
  if (!MBOrErr) return false;
  out = MBOrErr->get()->getBuffer().str();
  return true;
}

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
  std::string content;
  if (!readFileToString(mainSourceAbsPath, content)) {
    errs() << "\n" << sHipify << sError << "Cannot read source file: " << mainSourceAbsPath << "\n";
    return false;
  }

  std::regex includeLineRe(R"re(^\s*#\s*include\s*"([^"]+)"\s*(?:\/\/.*)?$)re", std::regex::ECMAScript);

  std::smatch m;
  std::istringstream iss(content);
  std::string line;
  std::set<std::string> headersFound;

  SmallString<256> srcDirPath(mainSourceAbsPath);
  sys::path::remove_filename(srcDirPath);
  sys::path::remove_dots(srcDirPath, true);
  std::string srcDir = std::string(srcDirPath.str());

  while (std::getline(iss, line)) {
    if (std::regex_search(line, m, includeLineRe)) {
      std::string headerRel = m[1].str();

      SmallString<256> hdrPath(mainSourceAbsPath);
      sys::path::remove_filename(hdrPath);
      sys::path::append(hdrPath, headerRel);
      std::string hdrFull = normalizeSmallStringPath(hdrPath);

      if (pathExists(hdrFull)) {
        headersFound.insert(hdrFull);
      } else {
        SmallString<256> altPath(srcDir);
        sys::path::append(altPath, headerRel);
        sys::path::remove_dots(altPath, true);
        std::string altFull = std::string(altPath.str());
        if (pathExists(altFull)) headersFound.insert(altFull);
      }
    }
  }

  if (headersFound.empty()) return true;

  vector<string> toProcess(headersFound.begin(), headersFound.end());
  set<string> processed;

  while (!toProcess.empty()) {
    string hdr = toProcess.back();
    toProcess.pop_back();
    if (processed.count(hdr)) continue;
    processed.insert(hdr);

    bool ok = runHipifyOnSingleFile(hdr, mainSourceAbsPath, compDB, OptionsParserPtr, hipify_exe, false);
    if (!ok) {
      errs() << "\n" << sHipify << sError << "Hipify failed for header: " << hdr << "\n";
    }

    if (recursive) {
      string hdrContent;
      if (readFileToString(hdr, hdrContent)) {
        istringstream iss2(hdrContent);
        string line2;
        while (getline(iss2, line2)) {
          if (std::regex_match(line2, m, includeLineRe)) {
            string subRel = m[1].str();
            SmallString<256> subPath(hdr);
            sys::path::remove_filename(subPath);
            sys::path::append(subPath, subRel);
            sys::path::remove_dots(subPath, true);
            string subFull = std::string(subPath.str());
            if (pathExists(subFull) && !processed.count(subFull)) {
              toProcess.push_back(subFull);
            }
          }
        }
      }
    }
  }

  return true;
}
