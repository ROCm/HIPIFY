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
  static const std::regex LocalIncludeRe(
      R"(^\s*#\s*include\s*\"([^\"\n]+)\"\s*(?://.*)?$)", std::regex::ECMAScript);

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

  bool collectPrecedingIncludes(const std::string &mainSourceAbspath,
                                const std::string &targetHeaderAbspath,
                                std::vector<std::string> &outIncludes) {
    std::string mainSourceContent;
    if (!readFile(mainSourceAbspath, mainSourceContent)) {
      errs() << sHipify << sError << "Cannot read source files: "
            << mainSourceAbspath << "\n";
      return false;
    }

    std::string targetFileName = std::string(sys::path::filename(targetHeaderAbspath));
    std::istringstream iss(mainSourceContent);
    std::string line;
    std::smatch m;

    while (std::getline(iss, line)) {
      if (std::regex_match(line, m, LocalIncludeRe)) {
        std::string quotedName = m[1].str();
        std::string quotedFileName = std::string(sys::path::filename(quotedName));
        if (quotedFileName == targetFileName)
          break;
        std::string absPath;
        if (resolveLocalIncludeInternal(mainSourceAbspath, quotedName, absPath))
          outIncludes.push_back(absPath);
      }

      if (std::regex_search(line, m, SystemIncludeRe)) {
        outIncludes.push_back(m[1].str());
      }
    }

    return true;
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

  std::vector<std::string> work(initial.begin(), initial.end());
  std::set<std::string> processed;
  size_t total = initial.size();
  size_t current = 0;

  while (!work.empty()) {
    std::string hdr = work.back();
    work.pop_back();
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
    collectPrecedingIncludes(mainSourceAbsPath, hdr, precedingIncludes);

    outs() << "\n" << sHipify << "Hipifying local header [" << current
           << "/" << total << "]: " << sys::path::filename(hdr) << "\n";

    bool ok = hipifySingleSource(hdr, hipOut, compDB, OptionsParserPtr,
                                  hipify_exe, mainSourceAbsPath, false,
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
        if (std::regex_match(line, m, LocalIncludeRe)) {
          std::string rel = m[1].str();
          std::string abs;
          if (resolveLocalIncludeInternal(hdr, rel, abs) &&
              !processed.count(abs)) {
            work.push_back(abs);
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
