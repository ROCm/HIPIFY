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

#include <memory>
#include <string>
#include <vector>

#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

namespace {

std::string getFilePathForID(const clang::SourceManager &SM,
                             clang::FileID FID) {
  const clang::FileEntry *FE = SM.getFileEntryForID(FID);
  if (!FE)
    return std::string();
  StringRef RealPath = FE->tryGetRealPathName();
  if (!RealPath.empty())
    return RealPath.str();
  return SM.getFilename(SM.getLocForStartOfFile(FID)).str();
}

// Records every `#include` of a translation unit, except those issued from a
// system header or from the command line.
class IncludeCollectorCallbacks : public clang::PPCallbacks {
  const clang::SourceManager &SM;
  std::vector<IncludeEntry> &Entries;

public:
  IncludeCollectorCallbacks(const clang::SourceManager &SM,
                            std::vector<IncludeEntry> &Entries)
      : SM(SM), Entries(Entries) {}

  void InclusionDirective(clang::SourceLocation HashLoc, const clang::Token &,
                          StringRef FileName, bool IsAngled,
                          clang::CharSourceRange,
#if LLVM_VERSION_MAJOR < 15
                          const clang::FileEntry *File,
#elif LLVM_VERSION_MAJOR == 15
                          Optional<clang::FileEntryRef> File,
#else
                          clang::OptionalFileEntryRef File,
#endif
                          StringRef, StringRef,
#if LLVM_VERSION_MAJOR < 19
                          const clang::Module *
#else
                          const clang::Module *, bool
#endif
#if LLVM_VERSION_MAJOR > 6
                          ,
                          clang::SrcMgr::CharacteristicKind FileType
#endif
                          ) override {
    const clang::FileID IncluderID = SM.getFileID(HashLoc);
    std::string IncluderPath = getFilePathForID(SM, IncluderID);
    if (IncluderPath.empty() || SM.isInSystemHeader(HashLoc))
      return;

    IncludeEntry Entry;
    Entry.fileName = FileName.str();
    Entry.isAngled = IsAngled;
    Entry.includerPath = std::move(IncluderPath);
    Entry.isFromMainFile = IncluderID == SM.getMainFileID();
#if LLVM_VERSION_MAJOR > 6
    Entry.isSystem = FileType != clang::SrcMgr::C_User;
#endif

    if (File) {
#if LLVM_VERSION_MAJOR < 15
      Entry.resolvedPath = File->tryGetRealPathName().str();
      if (Entry.resolvedPath.empty())
        Entry.resolvedPath = File->getName().str();
#else
      Entry.resolvedPath = File->getFileEntry().tryGetRealPathName().str();
      if (Entry.resolvedPath.empty())
        Entry.resolvedPath = File->getName().str();
#endif
    }

    Entries.push_back(std::move(Entry));
  }
};

// PreprocessOnlyAction supplies the lexing loop and IgnorePragmas().
class IncludeCollectorAction : public clang::PreprocessOnlyAction {
  std::vector<IncludeEntry> &Entries;

public:
  explicit IncludeCollectorAction(std::vector<IncludeEntry> &Entries)
      : Entries(Entries) {}

protected:
#if LLVM_VERSION_MAJOR < 5
  bool BeginSourceFileAction(clang::CompilerInstance &CI, StringRef) override {
#else
  bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
#endif
    CI.getPreprocessor().addPPCallbacks(
        std::make_unique<IncludeCollectorCallbacks>(CI.getSourceManager(),
                                                    Entries));
    return true;
  }
};

class IncludeCollectorActionFactory : public FrontendActionFactory {
  std::vector<IncludeEntry> &Entries;

public:
  explicit IncludeCollectorActionFactory(std::vector<IncludeEntry> &entries)
      : Entries(entries) {}

#if LLVM_VERSION_MAJOR >= 10
  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<IncludeCollectorAction>(Entries);
  }
#else
  clang::FrontendAction *create() override {
    return new IncludeCollectorAction(Entries);
  }
#endif
};

// Returns entries.size() when headerPath was never included.
size_t findFirstInclusion(const std::vector<IncludeEntry> &entries,
                          StringRef headerPath) {
  size_t i = 0;
  for (; i < entries.size(); ++i)
    if (entries[i].resolvedPath == headerPath)
      break;
  return i;
}

// Files to `-include` in front of headerPath so that a header which is not
// self-contained still sees what its ancestors included before it.
std::vector<std::string>
buildIncludeContext(const std::vector<IncludeEntry> &entries,
                    StringRef headerPath) {
  const size_t pos = findFirstInclusion(entries, headerPath);
  if (pos == entries.size())
    return std::vector<std::string>();

  StringSet<> ancestors;
  std::string file = entries[pos].includerPath;
  while (!file.empty() && ancestors.insert(file).second) {
    const size_t idx = findFirstInclusion(entries, file);
    file = idx == entries.size() ? std::string() : entries[idx].includerPath;
  }

  std::vector<std::string> context;
  StringSet<> seen;
  for (size_t i = 0; i < pos; ++i) {
    const IncludeEntry &e = entries[i];
    // An ancestor would re-include headerPath, whose guard then empties the
    // copy being hipified.
    if (!ancestors.count(e.includerPath) || ancestors.count(e.resolvedPath))
      continue;
    // System headers are injected as spelled, to be found via the search paths.
    std::string arg = e.isSystem ? e.fileName : e.resolvedPath;
    if (arg.empty() || !seen.insert(arg).second)
      continue;
    context.push_back(std::move(arg));
  }
  return context;
}

} // namespace

bool appendArgumentsAdjusters(ct::RefactoringTool &Tool,
                              const std::string &sSourceAbsPath,
                              const char *hipify_exe);

bool collectIncludeTree(const std::string &srcPath,
                        const ct::CompilationDatabase *compDB,
                        ct::CommonOptionsParser *OptionsParserPtr,
                        const char *hipify_exe,
                        std::vector<IncludeEntry> &outEntries) {
  outEntries.clear();

  ct::RefactoringTool Tool(
      compDB ? *compDB : OptionsParserPtr->getCompilations(), {srcPath});

  if (!appendArgumentsAdjusters(Tool, srcPath, hipify_exe)) {
    return false;
  }

  IncludeCollectorActionFactory factory(outEntries);
  return Tool.run(&factory) == 0;
}

bool hipifyLocalHeaders(const std::string &mainSourceAbsPath,
                        const ct::CompilationDatabase *compDB,
                        ct::CommonOptionsParser *OptionsParserPtr,
                        const char *hipify_exe, bool recursive) {
  std::vector<IncludeEntry> entries;
  if (!collectIncludeTree(mainSourceAbsPath, compDB, OptionsParserPtr,
                          hipify_exe, entries)) {
    errs() << "\n"
           << sHipify << sError
           << "Failed to collect includes from: " << mainSourceAbsPath << "\n";
    return false;
  }

  // The single run already walked the whole tree; recursion is just the
  // unfiltered result. Include guards make each path appear at most once.
  std::vector<std::string> headers;
  StringSet<> seen;
  for (const IncludeEntry &e : entries) {
    if (e.isAngled || e.isSystem || e.resolvedPath.empty())
      continue;
    if (!recursive && !e.isFromMainFile)
      continue;
    if (seen.insert(e.resolvedPath).second)
      headers.push_back(e.resolvedPath);
  }

  if (headers.empty()) {
    outs() << "\n" << sHipify << "No local headers detected in "
           << sys::path::filename(mainSourceAbsPath) << "\n";
    return true;
  }

  outs() << "\n" << sHipify << "Local headers found: " << headers.size()
         << " in " << sys::path::filename(mainSourceAbsPath) << "\n";
  for (size_t i = 0; i < headers.size(); ++i) {
    outs() << (i + 1) << "/" << headers.size() << ": "
           << sys::path::filename(headers[i]) << "\n";
  }

  for (size_t i = 0; i < headers.size(); ++i) {
    const std::string &hdr = headers[i];
    std::string hipOut = hdr + ".hip";
    if (!hipifySingleSource(hdr, hipOut, compDB, OptionsParserPtr, hipify_exe,
                            mainSourceAbsPath, false,
                            buildIncludeContext(entries, hdr))) {
      errs() << "\n" << sHipify << sError << "Hipify failed for header ["
             << (i + 1) << "/" << headers.size() << "]: "
             << sys::path::filename(hdr) << "\n";
      return false;
    }
    outs() << sHipify << "Successfully hipified header file" << "\n";
  }

  outs() << "\n" << sHipify << "Local header hipification complete: "
         << headers.size() << " header(s) processed.\n";
  return true;
}
