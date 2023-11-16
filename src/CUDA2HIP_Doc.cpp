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
  typedef map<StringRef, hipAPIversions> hipVersionMap;
  typedef map<llvm::StringRef, hipAPIChangedVersions> hipChangedVersionMap;
  typedef map<llvm::StringRef, cudaAPIChangedVersions> cudaChangedVersionMap;

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

  const string sandROC = "_and_ROC";
  const string sBLAS = "CUBLAS_API_supported_by_HIP";
  const string sBLAS_md = sBLAS + md_ext;
  const string sBLAS_csv = sBLAS + csv_ext;
  const string sBLAS_and_ROC_md = sBLAS + sandROC + md_ext;
  const string sBLAS_and_ROC_csv = sBLAS + sandROC + csv_ext;
  const string sROCBLAS = "CUBLAS_API_supported_by_ROC";
  const string sROCBLAS_md = sROCBLAS + md_ext;
  const string sROCBLAS_csv = sROCBLAS + csv_ext;
  const string sCUBLAS = "CUBLAS";

  const string sSOLVER = "CUSOLVER_API_supported_by_HIP";
  const string sSOLVER_md = sSOLVER + md_ext;
  const string sSOLVER_csv = sSOLVER + csv_ext;
  const string sSOLVER_and_ROC_md = sSOLVER + sandROC + md_ext;
  const string sSOLVER_and_ROC_csv = sSOLVER + sandROC + csv_ext;
  const string sROCSOLVER = "CUSOLVER_API_supported_by_ROC";
  const string sROCSOLVER_md = sROCSOLVER + md_ext;
  const string sROCSOLVER_csv = sROCSOLVER + csv_ext;
  const string sCUSOLVER = "CUSOLVER";

  const string sRAND = "CURAND_API_supported_by_HIP";
  const string sRAND_md = sRAND + md_ext;
  const string sRAND_csv = sRAND + csv_ext;
  const string sCURAND = "CURAND";

  const string sandMIOPEN = "_and_MIOPEN";
  const string sDNN = "CUDNN_API_supported_by_HIP";
  const string sDNN_md = sDNN + md_ext;
  const string sDNN_csv = sDNN + csv_ext;
  const string sDNN_and_MIOPEN_md = sDNN + sandMIOPEN + md_ext;
  const string sDNN_and_MIOPEN_csv = sDNN + sandMIOPEN + csv_ext;
  const string sMIOPEN_ = "CUDNN_API_supported_by_MIOPEN";
  const string sMIOPEN_md = sMIOPEN_ + md_ext;
  const string sMIOPEN_csv = sMIOPEN_ + csv_ext;
  const string sCUDNN = "CUDNN";

  const string sFFT = "CUFFT_API_supported_by_HIP";
  const string sFFT_md = sFFT + md_ext;
  const string sFFT_csv = sFFT + csv_ext;
  const string sCUFFT = "CUFFT";

  const string sSPARSE = "CUSPARSE_API_supported_by_HIP";
  const string sSPARSE_md = sSPARSE + md_ext;
  const string sSPARSE_csv = sSPARSE + csv_ext;
  const string sSPARSE_and_ROC_md = sSPARSE + sandROC + md_ext;
  const string sSPARSE_and_ROC_csv = sSPARSE + sandROC + csv_ext;
  const string sROCSPARSE = "CUSPARSE_API_supported_by_ROC";
  const string sROCSPARSE_md = sROCSPARSE + md_ext;
  const string sROCSPARSE_csv = sROCSPARSE + csv_ext;
  const string sCUSPARSE = "CUSPARSE";

  const string sDEVICE = "CUDA_Device_API_supported_by_HIP";
  const string sDEVICE_md = sDEVICE + md_ext;
  const string sDEVICE_csv = sDEVICE + csv_ext;
  const string sCUDEVICE = "CUDA DEVICE";

  const string sRTC = "CUDA_RTC_API_supported_by_HIP";
  const string sRTC_md = sRTC + md_ext;
  const string sRTC_csv = sRTC + csv_ext;
  const string sCURTC = "CUDA RTC";

  const string sCUB = "CUB_API_supported_by_HIP";
  const string sCUB_md = sCUB + md_ext;
  const string sCUB_csv = sCUB + csv_ext;
  const string sCUCUB = "CUB";

  const string sAPI_supported_by = "API supported by ";
  const string sCUDA = "CUDA";
  const string sHIP = "HIP";
  const string sROC = "ROC";
  const string sMIOPEN = "MIOPEN";
  const string sHIPandROC = "HIP and ROC";
  const string sHIPandMIOPEN = "HIP and MIOPEN";
  const string sA = "A";
  const string sD = "D";
  const string sC = "C";
  const string sR = "R";
  const string sE = "E";
  const hipChangedVersionMap hipChangedVersionMapEmpty = {};
  const cudaChangedVersionMap cudaChangedVersionMapEmpty = {};

  enum docType {
    none = 0,
    md = 1,
    csv = 2,
  };

  enum docFormat {
    full = 0,
    strict = 1,
    compact = 2,
  };

  enum docRoc {
    skip = 0,
    separate = 1,
    joint = 2,
  };

  class DOC {
    public:
      DOC(const string &outDir): dir(outDir), types(0), format(0), hasROC(false), isROC(false) {}
      virtual ~DOC() {}
      void setTypesAndFormat(unsigned int docTypes, unsigned int docFormat = full, unsigned int docRoc = skip) {
        types = docTypes;
        format = docFormat;
        roc = docRoc;
      }
      bool generate() {
        if (init()) return write() & fini();
        return false;
      }
      virtual void setCommonHipVersionMap() {}

    protected:
      virtual const string &getFileName(docType t) const = 0;
      virtual const string &getName() const = 0;
      virtual const sectionMap &getSections() const = 0;
      virtual const functionMap &getFunctions() const = 0;
      virtual const typeMap &getTypes() const = 0;
      virtual const versionMap &getFunctionVersions() const = 0;
      virtual const hipVersionMap &getHipFunctionVersions() const = 0;
      virtual const hipChangedVersionMap &getHipChangedFunctionVersions() const { return hipChangedVersionMapEmpty; };
      virtual const cudaChangedVersionMap &getCudaChangedFunctionVersions() const { return cudaChangedVersionMapEmpty; };
      virtual const versionMap &getTypeVersions() const = 0;
      virtual const hipVersionMap &getHipTypeVersions() const = 0;
      virtual const string &getAPI() const { return sHIP; }
      virtual const string &getSecondAPI() const { return sROC; }
      virtual const string &getJointAPI() const { return sEmpty; }
      virtual bool isJoint() const { return roc == joint && hasROC; }
      hipVersionMap commonHipVersionMap;
      bool hasROC;
      bool isROC;
      unsigned int roc;

    private:
      string dir;
      error_code EC;
      unsigned int types;
      unsigned int format;
      map<docType, string> files;
      map<docType, string> tmpFiles;
      map<docType, unique_ptr<ostream>> streams;

      bool init(docType t) {
        string file = (dir.empty() ? getFileName(t) : dir + "/" + getFileName(t));
        SmallString<128> tmpFile;
        EC = sys::fs::createTemporaryFile(file, getExtension(t), tmpFile);
        if (EC) {
          errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
          return false;
        }
        files.insert({ t, file });
        tmpFiles.insert({ t, tmpFile.str().str() });
        streams.insert(make_pair(t, unique_ptr<ostream>(new ofstream(tmpFile.c_str(), ios_base::trunc))));
        return true;
      }

      bool init() {
        bool bRet = true;
        if (md == (types & md)) bRet = init(md);
        if (csv == (types & csv)) bRet = init(csv) & bRet;
        return bRet;
      }

      bool isTypeSection(unsigned int n, const sectionMap &sections) {
        string name = string(sections.at(n));
        for (auto &c : name) c = tolower(c);
        return name.find("type") != string::npos;
      }

      bool write() {
        const docType docs[] = {md, csv};
        for (auto doc : docs) {
          if (doc != (types & doc)) continue;
          *streams[doc].get() << (doc == md ? "# " : "") << getName() << " " << sAPI_supported_by << (isJoint() ? getJointAPI() : getAPI()) << endl << endl;
          unsigned int compact_only_cur_sec_num = 1;
          for (auto &s : getSections()) {
            const functionMap &ftMap = isTypeSection(s.first, getSections()) ? getTypes() : getFunctions();
            const versionMap &vMap = isTypeSection(s.first, getSections()) ? getTypeVersions() : getFunctionVersions();
            const hipVersionMap &hMap = commonHipVersionMap.empty() ? (isTypeSection(s.first, getSections()) ? getHipTypeVersions() : getHipFunctionVersions()) : commonHipVersionMap;
            const hipChangedVersionMap &hChangedMap = getHipChangedFunctionVersions();
            const cudaChangedVersionMap &cudaChangedMap = getCudaChangedFunctionVersions();
            functionMap fMap;
            for (auto &f : ftMap) {
              if (f.second.apiSection == s.first) {
                if (isROC) {
                  if (format != compact || (format == compact && !Statistics::isRocUnsupported(f.second))) {
                    fMap.insert(f);
                  }
                } else {
                  if (format != compact || (format == compact && ((!isJoint() && !Statistics::isHipUnsupported(f.second)) ||
                                                                   (isJoint() && (!Statistics::isHipUnsupported(f.second) || !Statistics::isRocUnsupported(f.second)))))) {
                    fMap.insert(f);
                  }
                }
              }
            }
            string sS = (doc == md) ? "|" : ",";
            stringstream rows;
            for (auto &f : fMap) {
              string a, d, c, r, ha, hd, hc, hr, he, ra, rd, rc, rr, re, cc;
              for (auto &v : vMap) {
                if (v.first == f.first) {
                  a = Statistics::getCudaVersion(v.second.appeared);
                  d = Statistics::getCudaVersion(v.second.deprecated);
                  r = Statistics::getCudaVersion(v.second.removed);
                  break;
                }
              }
              auto cudacv = cudaChangedMap.find(f.first);
              if (cudacv != cudaChangedMap.end()) {
                int icount = 0;
                for (auto &cv : cudacv->second) {
                  if (icount > 0)
                    c += ", ";
                  c += Statistics::getCudaVersion(cv);
                  icount++;
                }
              }
              auto hv = (isROC) ? hMap.find(f.second.rocName) : hMap.find(f.second.hipName);
              if (hv != hMap.end() && ((!Statistics::isRocUnsupported(f.second) && isROC) || (!Statistics::isHipUnsupported(f.second) && !isROC))) {
                ha = Statistics::getHipVersion(hv->second.appeared);
                hd = Statistics::getHipVersion(hv->second.deprecated);
                hr = Statistics::getHipVersion(hv->second.removed);
                he = Statistics::getHipVersion(hv->second.experimental);
              }
              auto hcv = (isROC) ? hChangedMap.find(f.second.rocName) : hChangedMap.find(f.second.hipName);
              if (hcv != hChangedMap.end() && ((!Statistics::isRocUnsupported(f.second) && isROC) || (!Statistics::isHipUnsupported(f.second) && !isROC))) {
                int icount = 0;
                for (auto &cv : hcv->second) {
                  if (icount > 0)
                    cc += ", ";
                  cc += Statistics::getHipVersion(cv);
                  icount++;
                }
                if (isROC)
                  rc = cc;
                else
                  hc = cc;
              }
              if (isJoint()) {
                auto rv = hMap.find(f.second.rocName);
                if (rv != hMap.end() && !Statistics::isRocUnsupported(f.second)) {
                  ra = Statistics::getHipVersion(rv->second.appeared);
                  rd = Statistics::getHipVersion(rv->second.deprecated);
                  rr = Statistics::getHipVersion(rv->second.removed);
                  re = Statistics::getHipVersion(rv->second.experimental);
                }
                auto hrv = hChangedMap.find(f.second.rocName);
                if (hrv != hChangedMap.end() && !Statistics::isRocUnsupported(f.second)) {
                  int icount = 0;
                  for (auto &cv : hrv->second) {
                    if (icount > 0)
                      rc += ", ";
                    rc += Statistics::getHipVersion(cv);
                    icount++;
                  }
                }
              }
              string sHip, sRoc;
              if (isROC)
                sHip = Statistics::isRocUnsupported(f.second) ? "" : string(f.second.rocName);
              else {
                sHip = Statistics::isHipUnsupported(f.second) ? "" : string(f.second.hipName);
                if (isJoint())
                  sRoc = Statistics::isRocUnsupported(f.second) ? "" : string(f.second.rocName);
              }
              if (doc == md) {
                sHip = sHip.empty() ? " " : "`" + sHip + "`";
                if (isJoint())
                  sRoc = sRoc.empty() ? " " : "`" + sRoc + "`";
              }
              rows << (doc == md ? "|`" : "") << string(f.first) << (doc == md ? "`|" : sS);
              switch (doc) {
                case csv:
                  switch (format) {
                    case strict:
                    case compact:
                      rows << (d.empty() ? "" : "+") << sS << sHip << sS << (hd.empty() ? "" : "+") << sS << (he.empty() ? "" : "+");
                      if (isJoint())
                        rows << sS << sRoc << sS << (rd.empty() ? "" : "+") << sS << (re.empty() ? "" : "+");
                      rows << endl;
                      break;
                    case full:
                    default:
                      rows << a << sS << d << sS << c << sS << r << sS << sHip << sS << ha << sS << hd << sS << (isROC ? rc : hc) << sS << hr << sS << he;
                      if (isJoint())
                        rows << sS << sRoc << sS << ra << sS << rd << sS << rc << sS << rr << sS << re;
                      rows << endl;
                      break;
                  }
                  break;
                case md:
                default:
                  switch (format) {
                    case strict:
                    case compact:
                      rows << (d.empty() ? " " : "+") << sS << sHip << sS << (hd.empty() ? " " : "+") << sS << (he.empty() ? " " : "+") << sS;
                      if (isJoint())
                        rows << sRoc << sS << (rd.empty() ? " " : "+") << sS << (re.empty() ? " " : "+") << sS;
                      rows << endl;
                      break;
                    case full:
                    default:
                      rows << (a.empty() ? " " : a) << sS << (d.empty() ? " " : d) << sS << (c.empty() ? " " : c) << sS << (r.empty() ? " " : r) << sS << sHip << sS <<
                        (ha.empty() ? " " : ha) << sS << (hd.empty() ? " " : hd) << sS << (isROC ? (rc.empty() ? " " : rc) : (hc.empty() ? " " : hc)) << sS << (hr.empty() ? " " : hr) << sS << (he.empty() ? " " : he) << sS;
                      if (isJoint())
                        rows << sRoc << sS << (ra.empty() ? " " : ra) << sS << (rd.empty() ? " " : rd) << sS << (rc.empty() ? " " : rc) << sS << (rr.empty() ? " " : rr) << sS << (re.empty() ? " " : re) << sS;
                      rows << endl;
                      break;
                  }
                  break;
              }
            }
            sS = (doc == md) ? "**|**" : ",";
            stringstream section, section_header;
            section_header << (doc == md ? "## **" : "") << (format != compact ? s.first : compact_only_cur_sec_num) << ". " << string(s.second) << (doc == md ? "**" : "") << endl << endl;
            section << (doc == md ? "|**" : "") << sCUDA << sS << (format == full ? sA : "") << (format == full ? sS : "") <<
              sD << sS << (format == full ? sC : "") << (format == full ? sS : "") << (format == full ? sR : "") << (format == full ? sS : "") << getAPI() << sS << (format == full ? sA : "") << (format == full ? sS : "") <<
              sD << (format == full ? sS : "") << (format == full ? sC : "") << (format == full ? sS : "") << (format == full ? sR : "") << sS << sE;
            if (isJoint())
              section << sS << getSecondAPI() << sS << (format == full ? sA : "") << (format == full ? sS : "") << sD << (format == full ? sS : "") << (format == full ? sC : "") << (format == full ? sS : "") << (format == full ? sR : "") << sS << sE;
            section << (doc == md ? "**|" : "") << endl;
            if (doc == md) {
              section << "|:--|" << (format == full ? ":-:|" : "") << ":-:|" << (format == full ? ":-:|" : "") << (format == full ? ":-:|" : "") <<
                ":--|" << (format == full ? ":-:|" : "") << ":-:|" << (format == full ? ":-:|" : "") << (format == full ? ":-:|" : "") << ":-:|";
              if (isJoint())
                section << ":--|" << (format == full ? ":-:|" : "") << ":-:|" << (format == full ? ":-:|" : "") << (format == full ? ":-:|" : "") << ":-:|";
              section << endl;
            }
            switch (format) {
              case full:
              case strict:
              default:
                *streams[doc].get() << section_header.str();
                *streams[doc].get() << (fMap.empty() ? "Unsupported\n\n" : section.str());
                break;
              case compact:
                if (!rows.str().empty()) {
                  *streams[doc].get() << section_header.str() << section.str();
                  compact_only_cur_sec_num++;
                }
                break;
            }
            if (!rows.str().empty()) {
              *streams[doc].get() << rows.str() << endl;
            }
          }
          *streams[doc].get() << endl << (doc == md ? "\\" : "") << (format == full ? "*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental" : "*D - Deprecated; E - Experimental");
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
        if (md == (types & md)) bRet = fini(md);
        if (csv == (types & csv)) bRet = fini(csv) & bRet;
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
      unsigned int types;
      unsigned int format;
      unsigned int roc;
    public:
      DOCS(unsigned int docTypes, unsigned int docFormat, unsigned docRoc): types(docTypes), format(docFormat), roc(docRoc) {}
      virtual ~DOCS() {}
      void addDoc(DOC *doc) { docs.push_back(doc); doc->setTypesAndFormat(types, format, roc); }
      bool generate() {
        bool bRet = true;
        for (auto &d : docs) {
          d->setCommonHipVersionMap();
          bRet = d->generate() & bRet;
        }
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
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_DRIVER_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_DRIVER_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_DRIVER_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUDA_DRIVER; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sDRIVER_md;
          case csv: return sDRIVER_csv;
        }
      }
      void setCommonHipVersionMap() override {
        commonHipVersionMap.insert(HIP_DRIVER_FUNCTION_VER_MAP.begin(), HIP_DRIVER_FUNCTION_VER_MAP.end());
        commonHipVersionMap.insert(HIP_DRIVER_TYPE_NAME_VER_MAP.begin(), HIP_DRIVER_TYPE_NAME_VER_MAP.end());
        commonHipVersionMap.insert(HIP_RUNTIME_FUNCTION_VER_MAP.begin(), HIP_RUNTIME_FUNCTION_VER_MAP.end());
        commonHipVersionMap.insert(HIP_RUNTIME_TYPE_NAME_VER_MAP.begin(), HIP_RUNTIME_TYPE_NAME_VER_MAP.end());
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
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_RUNTIME_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_RUNTIME_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_RUNTIME_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUDA_RUNTIME; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sRUNTIME_md;
          case csv: return sRUNTIME_csv;
        }
      }
      void setCommonHipVersionMap() override {
        commonHipVersionMap.insert(HIP_DRIVER_FUNCTION_VER_MAP.begin(), HIP_DRIVER_FUNCTION_VER_MAP.end());
        commonHipVersionMap.insert(HIP_DRIVER_TYPE_NAME_VER_MAP.begin(), HIP_DRIVER_TYPE_NAME_VER_MAP.end());
        commonHipVersionMap.insert(HIP_RUNTIME_FUNCTION_VER_MAP.begin(), HIP_RUNTIME_FUNCTION_VER_MAP.end());
        commonHipVersionMap.insert(HIP_RUNTIME_TYPE_NAME_VER_MAP.begin(), HIP_RUNTIME_TYPE_NAME_VER_MAP.end());
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
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_COMPLEX_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_COMPLEX_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_COMPLEX_TYPE_NAME_VER_MAP; }
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
      BLAS(const string &outDir): DOC(outDir) { hasROC = true; }
      virtual ~BLAS() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_BLAS_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_BLAS_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_BLAS_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_BLAS_FUNCTION_VER_MAP; }
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_BLAS_FUNCTION_VER_MAP; }
      const hipChangedVersionMap &getHipChangedFunctionVersions() const override { return HIP_BLAS_FUNCTION_CHANGED_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_BLAS_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_BLAS_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUBLAS; }
      const string &getSecondAPI() const override { return sROC; }
      const string &getJointAPI() const override { return sHIPandROC; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return roc == joint ? sBLAS_and_ROC_md : sBLAS_md;
          case csv: return roc == joint ? sBLAS_and_ROC_csv : sBLAS_csv;
        }
      }
  };

  class ROCBLAS: public BLAS {
    public:
      ROCBLAS(const string &outDir): BLAS(outDir) { hasROC = false; isROC = true; }
      virtual ~ROCBLAS() {}
    protected:
      const string &getAPI() const override { return sROC; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sROCBLAS_md;
          case csv: return sROCBLAS_csv;
        }
      }
  };

  class SOLVER : public DOC {
  public:
    SOLVER(const string& outDir) : DOC(outDir) { hasROC = true; }
    virtual ~SOLVER() {}
  protected:
    const sectionMap& getSections() const override { return CUDA_SOLVER_API_SECTION_MAP; }
    const functionMap& getFunctions() const override { return CUDA_SOLVER_FUNCTION_MAP; }
    const typeMap& getTypes() const override { return CUDA_SOLVER_TYPE_NAME_MAP; }
    const versionMap& getFunctionVersions() const override { return CUDA_SOLVER_FUNCTION_VER_MAP; }
    const hipVersionMap& getHipFunctionVersions() const override { return HIP_SOLVER_FUNCTION_VER_MAP; }
    const versionMap& getTypeVersions() const override { return CUDA_SOLVER_TYPE_NAME_VER_MAP; }
    const hipVersionMap& getHipTypeVersions() const override { return HIP_SOLVER_TYPE_NAME_VER_MAP; }
    const string& getName() const override { return sCUSOLVER; }
    const string& getSecondAPI() const override { return sROC; }
    const string& getJointAPI() const override { return sHIPandROC; }
    const string& getFileName(docType format) const override {
      switch (format) {
      case none:
      default: return sEmpty;
      case md: return roc == joint ? sSOLVER_and_ROC_md : sSOLVER_md;
      case csv: return roc == joint ? sSOLVER_and_ROC_csv : sSOLVER_csv;
      }
    }
  };

  class ROCSOLVER : public SOLVER {
  public:
    ROCSOLVER(const string& outDir) : SOLVER(outDir) { hasROC = false; isROC = true; }
    virtual ~ROCSOLVER() {}
  protected:
    const string& getAPI() const override { return sROC; }
    const string& getFileName(docType format) const override {
      switch (format) {
      case none:
      default: return sEmpty;
      case md: return sROCSOLVER_md;
      case csv: return sROCSOLVER_csv;
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
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_RAND_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_RAND_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_RAND_TYPE_NAME_VER_MAP; }
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
      DNN(const string &outDir): DOC(outDir) { hasROC = true; }
      virtual ~DNN() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_DNN_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_DNN_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_DNN_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_DNN_FUNCTION_VER_MAP; }
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_DNN_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_DNN_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_DNN_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUDNN; }
      const string &getSecondAPI() const override { return sMIOPEN; }
      const string &getJointAPI() const override { return sHIPandMIOPEN; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return roc == joint ? sDNN_and_MIOPEN_md : sDNN_md;
          case csv: return roc == joint ? sDNN_and_MIOPEN_csv : sDNN_csv;
        }
      }
  };

  class MIOPEN: public DNN {
    public:
      MIOPEN(const string &outDir): DNN(outDir) { hasROC = false; isROC = true; }
      virtual ~MIOPEN() {}
    protected:
      const string &getAPI() const override { return sMIOPEN; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sMIOPEN_md;
          case csv: return sMIOPEN_csv;
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
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_FFT_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_FFT_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_FFT_TYPE_NAME_VER_MAP; }
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
      SPARSE(const string &outDir): DOC(outDir) { hasROC = true; }
      virtual ~SPARSE() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_SPARSE_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_SPARSE_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_SPARSE_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_SPARSE_FUNCTION_VER_MAP; }
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_SPARSE_FUNCTION_VER_MAP; }
      const hipChangedVersionMap &getHipChangedFunctionVersions() const override { return HIP_SPARSE_FUNCTION_CHANGED_VER_MAP; }
      const cudaChangedVersionMap &getCudaChangedFunctionVersions() const override { return CUDA_SPARSE_FUNCTION_CHANGED_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_SPARSE_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_SPARSE_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUSPARSE; }
      const string &getSecondAPI() const override { return sROC; }
      const string &getJointAPI() const override { return sHIPandROC; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return roc == joint ? sSPARSE_and_ROC_md : sSPARSE_md;
          case csv: return roc == joint ? sSPARSE_and_ROC_csv : sSPARSE_csv;
        }
      }
  };

  class ROCSPARSE : public SPARSE {
  public:
    ROCSPARSE(const string &outDir) : SPARSE(outDir) { hasROC = false; isROC = true; }
    virtual ~ROCSPARSE() {}
  protected:
    const string &getAPI() const override { return sROC; }
    const string &getFileName(docType format) const override {
      switch (format) {
      case none:
      default: return sEmpty;
      case md: return sROCSPARSE_md;
      case csv: return sROCSPARSE_csv;
      }
    }
  };

   class DEVICE : public DOC {
    public:
      DEVICE(const string &outDir): DOC(outDir) {}
      virtual ~DEVICE() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_DEVICE_FUNCTION_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_DEVICE_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_DEVICE_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_DEVICE_FUNCTION_VER_MAP; }
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_DEVICE_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_DEVICE_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_DEVICE_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUDEVICE; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sDEVICE_md;
          case csv: return sDEVICE_csv;
        }
      }
  };

   class RTC : public DOC {
    public:
      RTC(const string &outDir): DOC(outDir) {}
      virtual ~RTC() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_RTC_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_RTC_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_RTC_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_RTC_FUNCTION_VER_MAP; }
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_RTC_FUNCTION_VER_MAP; }
      const cudaChangedVersionMap& getCudaChangedFunctionVersions() const override { return CUDA_RTC_FUNCTION_CHANGED_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_RTC_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_RTC_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCURTC; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sRTC_md;
          case csv: return sRTC_csv;
        }
      }
  };

   class CUB : public DOC {
    public:
      CUB(const string &outDir): DOC(outDir) {}
      virtual ~CUB() {}
    protected:
      const sectionMap &getSections() const override { return CUDA_CUB_API_SECTION_MAP; }
      const functionMap &getFunctions() const override { return CUDA_CUB_FUNCTION_MAP; }
      const typeMap &getTypes() const override { return CUDA_CUB_TYPE_NAME_MAP; }
      const versionMap &getFunctionVersions() const override { return CUDA_CUB_FUNCTION_VER_MAP; }
      const hipVersionMap &getHipFunctionVersions() const override { return HIP_CUB_FUNCTION_VER_MAP; }
      const versionMap &getTypeVersions() const override { return CUDA_CUB_TYPE_NAME_VER_MAP; }
      const hipVersionMap &getHipTypeVersions() const override { return HIP_CUB_TYPE_NAME_VER_MAP; }
      const string &getName() const override { return sCUCUB; }
      const string &getFileName(docType format) const override {
        switch (format) {
          case none:
          default: return sEmpty;
          case md: return sCUB_md;
          case csv: return sCUB_csv;
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
    unsigned int docFormat = full;
    if (!DocFormat.empty()) {
      if (DocFormat == "compact") docFormat = compact;
      else if (DocFormat == "strict") docFormat = strict;
      else if (DocFormat != "full") {
        llvm::errs() << "\n" << sHipify << sError << "Unsupported documentation format: '" << DocFormat << "'; supported formats: 'full', 'strict', 'compact'\n";
        return false;
      }
    }
    unsigned int docRoc = skip;
    if (!DocRoc.empty()) {
      if (DocRoc == "separate") docRoc = separate;
      else if (DocRoc == "joint") docRoc = joint;
      else if (DocRoc != "skip") {
        llvm::errs() << "\n" << sHipify << sError << "Unsupported ROC documentation format: '" << DocRoc << "'; supported formats: 'skip', 'separate', 'joint'\n";
        return false;
      }
    }
    unsigned int docTypes = 0;
    if (GenerateMD) docTypes |= md;
    if (GenerateCSV) docTypes |= csv;
    DOCS docs(docTypes, docFormat, docRoc);
    DRIVER driver(sOut);
    docs.addDoc(&driver);
    RUNTIME runtime(sOut);
    docs.addDoc(&runtime);
    COMPLEX complex(sOut);
    docs.addDoc(&complex);
    BLAS blas(sOut);
    docs.addDoc(&blas);
    ROCBLAS rocblas(sOut);
    SOLVER solver(sOut);
    docs.addDoc(&solver);
    ROCSOLVER rocsolver(sOut);
    MIOPEN miopen(sOut);
    ROCSPARSE rocsparse(sOut);
    if (docRoc == separate) {
      docs.addDoc(&rocblas);
      docs.addDoc(&miopen);
      docs.addDoc(&rocsparse);
      docs.addDoc(&rocsolver);
    }
    RAND rand(sOut);
    docs.addDoc(&rand);
    DNN dnn(sOut);
    docs.addDoc(&dnn);
    FFT fft(sOut);
    docs.addDoc(&fft);
    SPARSE sparse(sOut);
    docs.addDoc(&sparse);
    DEVICE device(sOut);
    docs.addDoc(&device);
    RTC rtc(sOut);
    docs.addDoc(&rtc);
    CUB cub(sOut);
    docs.addDoc(&cub);
    return docs.generate();
  }

}
