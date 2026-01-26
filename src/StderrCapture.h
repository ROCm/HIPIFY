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
#include <sstream>
#include <fstream>
#include <cstdio>
#include <mutex>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <io.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <windows.h>
  #define STDERR_FD _fileno(stderr)
  #define DUP(fd) _dup(fd)
  #define DUP2(fd1, fd2) _dup2(fd1, fd2)
  #define CLOSE(fd) _close(fd)
#else
  #include <unistd.h>
  #include <fcntl.h>
  #define STDERR_FD STDERR_FILENO
  #define DUP(fd) dup(fd)
  #define DUP2(fd1, fd2) dup2(fd1, fd2)
  #define CLOSE(fd) close(fd)
#endif

namespace hipify {

// Capture stderr output to a temp file.
class StderrCapture {
public:
  StderrCapture() : lock_(getMutex()), saved_stderr_(-1), temp_fd_(-1), active_(false) {
#ifdef _WIN32
    char tempDir[MAX_PATH];
    DWORD tempDirLen = GetTempPathA(MAX_PATH, tempDir);
    if (tempDirLen == 0 || tempDirLen > MAX_PATH) return;
    
    char tempFile[MAX_PATH];
    if (GetTempFileNameA(tempDir, "hip", 0, tempFile) == 0) return;
    
    tempFilePath_ = tempFile;
    
    temp_fd_ = _open(tempFile, _O_RDWR | _O_CREAT | _O_TRUNC, _S_IREAD | _S_IWRITE);
    if (temp_fd_ == -1) {
      DeleteFileA(tempFile);
      tempFilePath_.clear();
      return;
    }
#else
    char tmpPath[] = "/tmp/hipify_stderr_XXXXXX";
    temp_fd_ = mkstemp(tmpPath);
    if (temp_fd_ == -1) return;
    
    tempFilePath_ = tmpPath;
#endif

    saved_stderr_ = DUP(STDERR_FD);
    if (saved_stderr_ == -1) {
      CLOSE(temp_fd_);
      removeTempFile();
      temp_fd_ = -1;
      return;
    }
    
    if (DUP2(temp_fd_, STDERR_FD) != -1) {
      active_ = true;
    }
  }
  
  // Restores stderr and deletes temp file.
  ~StderrCapture() {
    restore();
    cleanup();
  }
  
  StderrCapture(const StderrCapture&) = delete;
  StderrCapture& operator=(const StderrCapture&) = delete;
  
  // Restores stderr to original state.
  void restore() {
    if (active_ && saved_stderr_ != -1) {
      fflush(stderr);
      DUP2(saved_stderr_, STDERR_FD);
      active_ = false;
    }
    if (saved_stderr_ != -1) {
      CLOSE(saved_stderr_);
      saved_stderr_ = -1;
    }
  }
  
  // Returns captured stderr content and restores stderr.
  std::string getCapturedOutput() {
    std::string content;
    restore();
    
    if (temp_fd_ != -1 && !tempFilePath_.empty()) {
      std::ifstream file(tempFilePath_);
      if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        content = buffer.str();
        file.close();
      }
    }
    return content;
  }
  
  bool isActive() const { return active_; }
  
private:
  static std::mutex& getMutex() {
    static std::mutex mtx;
    return mtx;
  }

  void removeTempFile() {
    if (!tempFilePath_.empty()) {
#ifdef _WIN32
      DeleteFileA(tempFilePath_.c_str());
#else
      unlink(tempFilePath_.c_str());
#endif
      tempFilePath_.clear();
    }
  }

  void cleanup() {
    if (temp_fd_ != -1) {
      CLOSE(temp_fd_);
      temp_fd_ = -1;
    }
    removeTempFile();
  }

  std::unique_lock<std::mutex> lock_;
  int saved_stderr_;
  int temp_fd_;
  bool active_;
  std::string tempFilePath_;
};

} // namespace hipify

#undef STDERR_FD
#undef DUP
#undef DUP2
#undef CLOSE
