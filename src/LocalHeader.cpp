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

#include <set>
#include <vector>

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

namespace {

class IncludeCollectorCallbacks : public clang::PPCallbacks {
  const clang::SourceManager &SM;
  std::vector<IncludeEntry> &Entries;

public:
  IncludeCollectorCallbacks(const clang::SourceManager &SM,
                            std::vector<IncludeEntry> &entries)
      : SM(SM), Entries(entries) {}

  void InclusionDirective(clang::SourceLocation hash_loc,
                          const clang::Token &include_token,
                          StringRef file_name, bool is_angled,
                          clang::CharSourceRange filename_range,
#if LLVM_VERSION_MAJOR < 15
                          const clang::FileEntry *file,
#elif LLVM_VERSION_MAJOR == 15
                          Optional<clang::FileEntryRef> file,
#else
                          clang::OptionalFileEntryRef file,
#endif
                          StringRef search_path, StringRef relative_path,
#if LLVM_VERSION_MAJOR < 19
                          const clang::Module *SuggestedModule
#else
                          const clang::Module *SuggestedModule,
                          bool ModuleImported
#endif
#if LLVM_VERSION_MAJOR > 6
                          ,
                          clang::SrcMgr::CharacteristicKind FileType
#endif
                          ) override {
    if (!SM.isWrittenInMainFile(hash_loc))
      return;

    IncludeEntry entry;
    entry.fileName = file_name.str();
    entry.isAngled = is_angled;

    if (file) {
#if LLVM_VERSION_MAJOR < 15
      entry.resolvedPath = file->tryGetRealPathName().str();
      if (entry.resolvedPath.empty())
        entry.resolvedPath = file->getName().str();
#else
      entry.resolvedPath = file->getFileEntry().tryGetRealPathName().str();
      if (entry.resolvedPath.empty())
        entry.resolvedPath = file->getName().str();
#endif
    }

    Entries.push_back(std::move(entry));
  }
};

class IncludeCollectorAction : public clang::PreprocessorFrontendAction {
  std::vector<IncludeEntry> &Entries;

public:
  explicit IncludeCollectorAction(std::vector<IncludeEntry> &entries)
      : Entries(entries) {}

  void ExecuteAction() override {
    clang::CompilerInstance &CI = getCompilerInstance();
    clang::Preprocessor &PP = CI.getPreprocessor();
    PP.addPPCallbacks(std::make_unique<IncludeCollectorCallbacks>(
        CI.getSourceManager(), Entries));

    PP.EnterMainSourceFile();
    clang::Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(clang::tok::eof));
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

} // namespace

bool collectIncludeTree(const std::string &srcPath,
                        const ct::CompilationDatabase *compDB,
                        ct::CommonOptionsParser *OptionsParserPtr,
                        const char *hipify_exe,
                        const std::string &mainContextPath,
                        std::vector<IncludeEntry> &outEntries) {
  outEntries.clear();

  const ct::CompilationDatabase &baseDB =
      compDB ? *compDB : OptionsParserPtr->getCompilations();

  // If srcPath has no entry in the compilation database, fall back to a
  // FixedCompilationDatabase rooted at the mainContextPath's directory so that
  // the tool doesn't skip the file.
  std::vector<ct::CompileCommand> cmds = baseDB.getCompileCommands(srcPath);
  std::unique_ptr<ct::FixedCompilationDatabase> fallbackDB;
  if (cmds.empty()) {
    std::string dir = sys::path::parent_path(mainContextPath).str();
    fallbackDB = std::make_unique<ct::FixedCompilationDatabase>(
        dir, std::vector<std::string>());
  }

  ct::RefactoringTool Tool(fallbackDB ? *fallbackDB : baseDB, {srcPath});

  if (!appendArgumentsAdjusters(Tool, mainContextPath, hipify_exe)) {
    return false;
  }

  // Strip the implicit CUDA header that appendArgumentsAdjusters adds —
  // not needed for include scanning and would pollute the results.
  Tool.appendArgumentsAdjuster(
      [](const ct::CommandLineArguments &Args, StringRef) {
        ct::CommandLineArguments filtered;
        for (size_t i = 0; i < Args.size(); ++i) {
          if (Args[i] == "-include" && i + 1 < Args.size() &&
              Args[i + 1] == "cuda_runtime.h") {
            ++i;
            continue;
          }
          filtered.push_back(Args[i]);
        }
        return filtered;
      });

  IncludeCollectorActionFactory factory(outEntries);
  Tool.run(&factory);
  return true;
}

bool collectLocalQuotedIncludes(const std::string &mainSourceAbsPath,
                                const ct::CompilationDatabase *compDB,
                                ct::CommonOptionsParser *OptionsParserPtr,
                                const char *hipify_exe,
                                std::vector<std::string> &outHeaders) {
  std::vector<IncludeEntry> entries;
  if (!collectIncludeTree(mainSourceAbsPath, compDB, OptionsParserPtr,
                          hipify_exe, mainSourceAbsPath, entries)) {
    errs() << "\n"
           << sHipify << sError
           << "Failed to collect includes from: " << mainSourceAbsPath << "\n";
    return false;
  }

  std::set<std::string> uniq;
  for (const auto &e : entries) {
    if (!e.isAngled && !e.resolvedPath.empty())
      uniq.insert(e.resolvedPath);
  }
  outHeaders.assign(uniq.begin(), uniq.end());
  return true;
}

bool hipifyLocalHeaders(const std::string &mainSourceAbsPath,
                        const ct::CompilationDatabase *compDB,
                        ct::CommonOptionsParser *OptionsParserPtr,
                        const char *hipify_exe, bool recursive) {

  std::vector<std::string> initial;
  if (!collectLocalQuotedIncludes(mainSourceAbsPath, compDB, OptionsParserPtr,
                                  hipify_exe, initial)) {
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
  std::set<std::string> queued(initial.begin(), initial.end());
  size_t total = initial.size();
  size_t current = 0;

  while (!work.empty()) {
    std::string hdr = work.back();
    work.pop_back();
    if (processed.count(hdr)) {
      continue;
    }
    processed.insert(hdr);
    ++current;

    std::string hipOut = hdr + ".hip";
    bool ok = hipifySingleSource(hdr, hipOut, compDB, OptionsParserPtr,
                                 hipify_exe, mainSourceAbsPath, false);

    if (!ok) {
      errs() << "\n" << sHipify << sError
             << "Hipify failed for header [" << current << "/" << total
             << "]: " << sys::path::filename(hdr) << "\n";
      return false;
    }
    outs() << sHipify << "Successfully hipified header file" << "\n";

    if (recursive) {
      std::vector<IncludeEntry> childEntries;
      if (collectIncludeTree(hdr, compDB, OptionsParserPtr, hipify_exe,
                             mainSourceAbsPath, childEntries)) {
        std::vector<std::string> newHeaders;
        for (const auto &e : childEntries) {
          if (!e.isAngled && !e.resolvedPath.empty() &&
              !processed.count(e.resolvedPath) &&
              !queued.count(e.resolvedPath)) {
            newHeaders.push_back(e.resolvedPath);
            work.push_back(e.resolvedPath);
            queued.insert(e.resolvedPath);
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
  }

  outs() << "\n" << sHipify << "Local header hipification complete: "
         << processed.size() << " header(s) processed.\n";
  return true;
}
