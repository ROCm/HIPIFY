/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <sstream>
#include <vector>
#include <map>
#include "llvm/ADT/StringRef.h"
#include "CUDA2HIP.h"
#include "CUDA2HIP_Scripting.h"
#include "ArgParse.h"
#include "StringUtils.h"
#include "LLVMCompat.h"

namespace doc {

  using namespace std;
  using namespace llvm;

  typedef map<unsigned int, llvm::StringRef> sectionMap;

  const string sEmpty = "";
  const string sMd = "md";
  const string md_ext = "." + sMd;
  const string sCsv = "csv";
  const string csv_ext = "." + sCsv;

  const string sDRIVER = "CUDA_Driver_API_functions_supported_by_HIP";
  const string sDRIVER_md = sDRIVER + md_ext;
  const string sDRIVER_csv = sDRIVER + csv_ext;
  const string sCUDA_DRIVER = "CUDA Driver";

  const string sRUNTIME = "CUDA_Runtime_API_functions_supported_by_HIP";
  const string sRUNTIME_md = sRUNTIME + md_ext;
  const string sRUNTIME_csv = sRUNTIME + csv_ext;
  const string sCUDA_RUNTIME = "CUDA Runtime";

  const string sCOMPLEX = "cuComplex_API_supported_by_HIP";
  const string sCOMPLEX_md = sCOMPLEX + md_ext;
  const string sCOMPLEX_csv = sCOMPLEX + csv_ext;
  const string sCUCOMPLEX = "CUCOMPLEX";

  const string sBLAS = "CUBLAS_API_supported_by_HIP";
  const string sBLAS_md = sBLAS + md_ext;
  const string sBLAS_csv = sBLAS + csv_ext;
  const string sCUBLAS = "CUBLAS";

  const string sRAND = "CURAND_API_supported_by_HIP";
  const string sRAND_md = sRAND + md_ext;
  const string sRAND_csv = sRAND + csv_ext;
  const string sCURAND = "CURAND";

  const string sDNN = "CUDNN_API_supported_by_HIP";
  const string sDNN_md = sDNN + md_ext;
  const string sDNN_csv = sDNN + csv_ext;
  const string sCUDNN = "CUDNN";

  const string sFFT = "CUFFT_API_supported_by_HIP";
  const string sFFT_md = sFFT + md_ext;
  const string sFFT_csv = sFFT + csv_ext;
  const string sCUFFT = "CUFFT";

  const string sSPARSE = "CUSPARSE_API_supported_by_HIP";
  const string sSPARSE_md = sSPARSE + md_ext;
  const string sSPARSE_csv = sSPARSE + csv_ext;
  const string sCUSPARSE = "CUSPARSE";

  const string sAPI_supported = "API supported by HIP";

  enum docType {
    none = 0,
    md = 1,
    csv = 2,
  };

  class DOC {
    public:
      DOC(const string &outDir): dir(outDir), formats(0) {}
      virtual ~DOC() {}
      void setFormats(unsigned int docFormats) { formats = docFormats; }
      bool generate() {
        if (init()) return write() & fini();
        return false;
      }

    protected:
      virtual const string &getFileName(docType format) const = 0;
      virtual const string &getName() const = 0;
      virtual const sectionMap &getSections() const = 0;

    private:
      string dir;
      error_code EC;
      unsigned int formats;
      map<docType, string> files;
      map<docType, string> tmpFiles;
      map<docType, unique_ptr<ostream>> streams;

      bool init(docType format) {
        string file = (dir.empty() ? getFileName(format) : dir + "/" + getFileName(format));
        SmallString<128> tmpFile;
        EC = sys::fs::createTemporaryFile(file, getExtension(format), tmpFile);
        if (EC) {
          llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
          return false;
        }
        files.insert({ format, file });
        tmpFiles.insert({ format, string(tmpFile) });
        streams.insert(make_pair(format, unique_ptr<ostream>(new ofstream(tmpFile.c_str(), ios_base::trunc))));
        return true;
      }

      bool init() {
        bool bRet = true;
        if (md == (formats & md)) bRet = init(md);
        if (csv == (formats & csv)) bRet = init(csv) & bRet;
        return bRet;
      }

      bool write() {
        if (md == (formats & md)) {
          *streams[md].get() << "# " << getName() << " " << sAPI_supported << endl << endl;
          for (auto &s : getSections()) {
            *streams[md].get() << "## **" << s.first << ". " << string(s.second) << "**" << endl << endl;
          }
        }
        return true;
      }

      bool fini(docType format) {
        streams[format].get()->flush();
        bool bRet = true;
        EC = sys::fs::copy_file(tmpFiles[format], files[format]);
        if (EC) {
          llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFiles[format] << " to " << files[format] << "\n";
          bRet = false;
        }
        if (!SaveTemps) sys::fs::remove(tmpFiles[format]);
        return bRet;
      }

      bool fini() {
        bool bRet = true;
        if (md == (formats & md)) bRet = fini(md);
        if (csv == (formats & csv)) bRet = fini(csv) & bRet;
        return bRet;
      }

      string getExtension(docType format) {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sMd;
          case csv: return sCsv;
        }
      }
  };

  class DOCS {
    private:
      vector<DOC*> docs;
      unsigned int formats;
    public:
      DOCS(unsigned int docFormats): formats(docFormats) {}
      virtual ~DOCS() {}
      void addDoc(DOC *doc) { docs.push_back(doc); doc->setFormats(formats); }
      bool generate() {
        bool bRet = true;
        for (auto &d : docs) bRet = d->generate() & bRet;
        return bRet;
      }
  };

  class DRIVER : public DOC {
    public:
      DRIVER(const string &outDir) : DOC(outDir) {}
      virtual ~DRIVER() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_DRIVER_API_SECTION_MAP; }
      const string &getName() const override { return sCUDA_DRIVER; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sDRIVER_md;
          case csv: return sDRIVER_csv;
        }
      }
  };

  class RUNTIME : public DOC {
    public:
      RUNTIME(const string &outDir): DOC(outDir) {}
      virtual ~RUNTIME() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_RUNTIME_API_SECTION_MAP; }
      const string &getName() const override { return sCUDA_RUNTIME; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sRUNTIME_md;
          case csv: return sRUNTIME_csv;
        }
      }
  };

  class COMPLEX : public DOC {
    public:
      COMPLEX(const string &outDir): DOC(outDir) {}
      virtual ~COMPLEX() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_COMPLEX_API_SECTION_MAP; }
      const string &getName() const override { return sCUCOMPLEX; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sCOMPLEX_md;
          case csv: return sCOMPLEX_csv;
        }
      }
  };

  class RAND: public DOC {
    public:
      RAND(const string &outDir): DOC(outDir) {}
      virtual ~RAND() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_RAND_API_SECTION_MAP; }
      const string &getName() const override { return sCURAND; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sRAND_md;
          case csv: return sRAND_csv;
        }
      }
  };

  class DNN: public DOC {
    public:
      DNN(const string &outDir): DOC(outDir) {}
      virtual ~DNN() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_DNN_API_SECTION_MAP; }
      const string &getName() const override { return sCUDNN; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sDNN_md;
          case csv: return sDNN_csv;
        }
      }
  };

  class FFT: public DOC {
    public:
      FFT(const string &outDir): DOC(outDir) {}
      virtual ~FFT() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_FFT_API_SECTION_MAP; }
      const string &getName() const override { return sCUFFT; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sFFT_md;
          case csv: return sFFT_csv;
        }
      }
  };

  class SPARSE: public DOC {
    public:
      SPARSE(const string &outDir): DOC(outDir) {}
      virtual ~SPARSE() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_SPARSE_API_SECTION_MAP; }
      const string &getName() const override { return sCUSPARSE; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sSPARSE_md;
          case csv: return sSPARSE_csv;
        }
      }
  };

  bool generate(bool GenerateMD, bool GenerateCSV) {
    if (!GenerateMD && !GenerateCSV) return true;
    std::error_code EC;
    std::string sOutputAbsPath = OutputDir;
    if (!sOutputAbsPath.empty()) {
      sOutputAbsPath = getAbsoluteDirectoryPath(sOutputAbsPath, EC, "documentation");
      if (EC) return false;
    }
    unsigned int docFormats = 0;
    if (GenerateMD) docFormats |= md;
    if (GenerateCSV) docFormats |= csv;
    DOCS docs(docFormats);
    DRIVER driver(sOutputAbsPath);
    docs.addDoc(&driver);
    RUNTIME runtime(sOutputAbsPath);
    docs.addDoc(&runtime);
    COMPLEX complex(sOutputAbsPath);
    docs.addDoc(&complex);
    RAND rand(sOutputAbsPath);
    docs.addDoc(&rand);
    DNN dnn(sOutputAbsPath);
    docs.addDoc(&dnn);
    FFT fft(sOutputAbsPath);
    docs.addDoc(&fft);
    SPARSE sparse(sOutputAbsPath);
    docs.addDoc(&sparse);
    return docs.generate();
  }

}
