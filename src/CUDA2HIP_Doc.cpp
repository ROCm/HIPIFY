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
#include "CUDA2HIP.h"
#include "CUDA2HIP_Scripting.h"
#include "ArgParse.h"
#include "StringUtils.h"
#include "LLVMCompat.h"

namespace doc {

  using namespace std;
  using namespace llvm;

  typedef map<unsigned int, StringRef> sectionMap;
  typedef map<StringRef, hipCounter> functionMap;
  typedef functionMap typeMap;
  typedef map<StringRef, cudaAPIversions> versionMap;

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
  const string sCUDA = "CUDA";
  const string sHIP = "HIP";
  const string sA = "A";
  const string sD = "D";
  const string sR = "R";

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
      virtual const functionMap &getFunctions() const = 0;
      virtual const typeMap &getTypes() const = 0;
      virtual const versionMap &getFunctionVersions() const = 0;
      virtual const versionMap &getTypeVersions() const = 0;

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
          errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
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

      bool isTypeSection(unsigned int n, const sectionMap &sections) {
        string name = string(sections.at(n));
        for (auto &c : name) c = tolower(c);
        return name.find("type") != string::npos;
      }

      bool write() {
        if (md == (formats & md)) {
          *streams[md].get() << "# " << getName() << " " << sAPI_supported << endl << endl;
          for (auto &s : getSections()) {
            *streams[md].get() << "## **" << s.first << ". " << string(s.second) << "**" << endl << endl;
            *streams[md].get() << "| **" << sCUDA << "** | **" << sA << "** | **" << sD << "** | **" << sR << "** | **" << sHIP << "** |" << endl;
            *streams[md].get() << "|:--|:-:|:-:|:-:|:--|" << endl;
            const functionMap &ftMap = isTypeSection(s.first, getSections()) ? getTypes() : getFunctions();
            const versionMap &vMap = isTypeSection(s.first, getSections()) ? getTypeVersions() : getFunctionVersions();
            functionMap fMap;
            for (auto &f : ftMap) if (f.second.apiSection == s.first) fMap.insert(f);
            for (auto &f : fMap) {
              string a, d, r;
              for (auto &v : vMap) {
                if (v.first == f.first) {
                  a = Statistics::getCudaVersion(v.second.appeared);
                  d = Statistics::getCudaVersion(v.second.deprecated);
                  r = Statistics::getCudaVersion(v.second.removed);
                }
              }
              *streams[md].get() << "|`" << string(f.first) << "`| " << a << " | " << d << " | " << r << " |" << (Statistics::isHipUnsupported(f.second) ? "" : "`" + string(f.second.hipName) + "`") << "|" << endl;
            }
            *streams[md].get() << endl;
          }
        }
        return true;
      }

      bool fini(docType format) {
        streams[format].get()->flush();
        bool bRet = true;
        EC = sys::fs::copy_file(tmpFiles[format], files[format]);
        if (EC) {
          errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFiles[format] << " to " << files[format] << "\n";
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
      const functionMap &getFunctions() const override { return CUDA_DRIVER_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_DRIVER_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_DRIVER_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_DRIVER_TYPE_NAME_VER_MAP; }
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
      const functionMap &getFunctions() const override { return CUDA_RUNTIME_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_RUNTIME_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_RUNTIME_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_RUNTIME_TYPE_NAME_VER_MAP; }
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
      const functionMap &getFunctions() const override { return CUDA_COMPLEX_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_COMPLEX_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_COMPLEX_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_COMPLEX_TYPE_NAME_VER_MAP; }
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

  class BLAS: public DOC {
    public:
      BLAS(const string &outDir): DOC(outDir) {}
      virtual ~BLAS() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_BLAS_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_BLAS_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_BLAS_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_BLAS_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_BLAS_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUBLAS; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sBLAS_md;
          case csv: return sBLAS_csv;
        }
      }
  };

  class RAND: public DOC {
    public:
      RAND(const string &outDir): DOC(outDir) {}
      virtual ~RAND() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_RAND_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_RAND_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_RAND_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_RAND_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_RAND_TYPE_NAME_VER_MAP; }
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
      const functionMap &getFunctions() const override { return CUDA_DNN_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_DNN_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_DNN_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_DNN_TYPE_NAME_VER_MAP; }
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
      const functionMap &getFunctions() const override { return CUDA_FFT_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_FFT_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_FFT_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_FFT_TYPE_NAME_VER_MAP; }
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
      const functionMap &getFunctions() const override { return CUDA_SPARSE_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_SPARSE_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_SPARSE_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_SPARSE_TYPE_NAME_VER_MAP; }
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
    error_code EC;
    string sOut = OutputDir;
    if (!sOut.empty()) {
      sOut = getAbsoluteDirectoryPath(sOut, EC, "documentation");
      if (EC) return false;
    }
    unsigned int docFormats = 0;
    if (GenerateMD) docFormats |= md;
    if (GenerateCSV) docFormats |= csv;
    DOCS docs(docFormats);
    DRIVER driver(sOut);
    docs.addDoc(&driver);
    RUNTIME runtime(sOut);
    docs.addDoc(&runtime);
    COMPLEX complex(sOut);
    docs.addDoc(&complex);
    BLAS blas(sOut);
    docs.addDoc(&blas);
    RAND rand(sOut);
    docs.addDoc(&rand);
    DNN dnn(sOut);
    docs.addDoc(&dnn);
    FFT fft(sOut);
    docs.addDoc(&fft);
    SPARSE sparse(sOut);
    docs.addDoc(&sparse);
    return docs.generate();
  }

}
