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

#include "CUDA2HIP.h"

const std::map<llvm::StringRef, hipCounter> CUDA_FILE_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  m["cufileop_status_error"]                                = {"hipFileOpStatusError",                         "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileHandleRegister"]                                 = {"hipFileHandleRegister",                        "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileHandleDeregister"]                               = {"hipFileHandleDeregister",                      "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileBufRegister"]                                    = {"hipFileBufRegister",                           "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileBufDeregister"]                                  = {"hipFileBufDeregister",                         "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileRead"]                                           = {"hipFileRead",                                  "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileWrite"]                                          = {"hipFileWrite",                                 "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverOpen"]                                     = {"hipFileDriverOpen",                            "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverClose"]                                    = {"hipFileDriverClose",                           "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverClose_v2"]                                 = {"hipFileDriverClose",                           "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileUseCount"]                                       = {"hipFileUseCount",                              "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverGetProperties"]                            = {"hipFileDriverGetProperties",                   "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverSetPollMode"]                              = {"hipFileDriverSetPollMode",                     "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverSetMaxDirectIOSize"]                       = {"hipFileDriverSetMaxDirectIOSize",              "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverSetMaxCacheSize"]                          = {"hipFileDriverSetMaxCacheSize",                 "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileDriverSetMaxPinnedMemSize"]                      = {"hipFileDriverSetMaxPinnedMemSize",             "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileWriteAsync"]                                     = {"hipFileWriteAsync",                            "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileReadAsync"]                                      = {"hipFileReadAsync",                             "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileStreamRegister"]                                 = {"hipFileStreamRegister",                        "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileStreamDeregister"]                               = {"hipFileStreamDeregister",                      "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileBatchIOSetUp"]                                   = {"hipFileBatchIOSetUp",                          "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileBatchIOSubmit"]                                  = {"hipFileBatchIOSubmit",                         "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileBatchIOGetStatus"]                               = {"hipFileBatchIOGetStatus",                      "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileBatchIOCancel"]                                  = {"hipFileBatchIOCancel",                         "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileBatchIODestroy"]                                 = {"hipFileBatchIODestroy",                        "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileGetParameterSizeT"]                              = {"hipFileGetParameterSizeT",                     "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileGetParameterBool"]                               = {"hipFileGetParameterBool",                      "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileGetParameterString"]                             = {"hipFileGetParameterString",                    "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileSetParameterSizeT"]                              = {"hipFileSetParameterSizeT",                     "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileSetParameterBool"]                               = {"hipFileSetParameterBool",                      "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileSetParameterString"]                             = {"hipFileSetParameterString",                    "", CONV_LIB_FUNC, API_FILE, 2};
  m["cuFileGetVersion"]                                     = {"hipFileGetVersion",                            "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileGetParameterMinMaxValue"]                        = {"hipFileGetParameterMinMaxValue",               "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileSetStatsLevel"]                                  = {"hipFileSetStatsLevel",                         "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileGetStatsLevel"]                                  = {"hipFileGetStatsLevel",                         "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileStatsStart"]                                     = {"hipFileStatsStart",                            "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileStatsStop"]                                      = {"hipFileStatsStop",                             "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileStatsReset"]                                     = {"hipFileStatsReset",                            "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileGetStatsL1"]                                     = {"hipFileGetStatsL1",                            "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileGetStatsL2"]                                     = {"hipFileGetStatsL2",                            "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileGetStatsL3"]                                     = {"hipFileGetStatsL3",                            "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileGetBARSizeInKB"]                                 = {"hipFileGetBARSizeInKB",                        "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileSetParameterPosixPoolSlabArray"]                 = {"hipFileSetParameterPosixPoolSlabArray",        "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileGetParameterPosixPoolSlabArray"]                 = {"hipFileGetParameterPosixPoolSlabArray",        "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileDriverGetP2PFlags"]                              = {"hipFileDriverGetP2PFlags",                     "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};
  m["cuFileDriverSetP2PFlags"]                              = {"hipFileDriverSetP2PFlags",                     "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_FILE_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cufileop_status_error"]                                = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileHandleRegister"]                                 = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileHandleDeregister"]                               = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileBufRegister"]                                    = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileBufDeregister"]                                  = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileRead"]                                           = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileWrite"]                                          = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileDriverOpen"]                                     = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileDriverClose"]                                    = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileDriverClose_v2"]                                 = {CUFILE_1040, CUDA_0, CUDA_0};
  m["cuFileUseCount"]                                       = {CUFILE_1040, CUDA_0, CUDA_0};
  m["cuFileDriverGetProperties"]                            = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileDriverSetPollMode"]                              = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileDriverSetMaxDirectIOSize"]                       = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileDriverSetMaxCacheSize"]                          = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileDriverSetMaxPinnedMemSize"]                      = {CUFILE_1000, CUDA_0, CUDA_0};
  m["cuFileWriteAsync"]                                     = {CUFILE_1070, CUDA_0, CUDA_0};
  m["cuFileReadAsync"]                                      = {CUFILE_1070, CUDA_0, CUDA_0};
  m["cuFileStreamRegister"]                                 = {CUFILE_1070, CUDA_0, CUDA_0};
  m["cuFileStreamDeregister"]                               = {CUFILE_1070, CUDA_0, CUDA_0};
  m["cuFileBatchIOSetUp"]                                   = {CUFILE_1020, CUDA_0, CUDA_0};
  m["cuFileBatchIOSubmit"]                                  = {CUFILE_1020, CUDA_0, CUDA_0};
  m["cuFileBatchIOGetStatus"]                               = {CUFILE_1020, CUDA_0, CUDA_0};
  m["cuFileBatchIOCancel"]                                  = {CUFILE_1020, CUDA_0, CUDA_0};
  m["cuFileBatchIODestroy"]                                 = {CUFILE_1020, CUDA_0, CUDA_0};
  m["cuFileGetParameterSizeT"]                              = {CUFILE_1140, CUDA_0, CUDA_0};
  m["cuFileGetParameterBool"]                               = {CUFILE_1140, CUDA_0, CUDA_0};
  m["cuFileGetParameterString"]                             = {CUFILE_1140, CUDA_0, CUDA_0};
  m["cuFileSetParameterSizeT"]                              = {CUFILE_1140, CUDA_0, CUDA_0};
  m["cuFileSetParameterBool"]                               = {CUFILE_1140, CUDA_0, CUDA_0};
  m["cuFileSetParameterString"]                             = {CUFILE_1140, CUDA_0, CUDA_0};
  m["cuFileGetVersion"]                                     = {CUFILE_1080, CUDA_0, CUDA_0};
  m["cuFileGetParameterMinMaxValue"]                        = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileSetStatsLevel"]                                  = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileGetStatsLevel"]                                  = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileStatsStart"]                                     = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileStatsStop"]                                      = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileStatsReset"]                                     = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileGetStatsL1"]                                     = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileGetStatsL2"]                                     = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileGetStatsL3"]                                     = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileGetBARSizeInKB"]                                 = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileSetParameterPosixPoolSlabArray"]                 = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileGetParameterPosixPoolSlabArray"]                 = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileDriverGetP2PFlags"]                              = {CUFILE_1150, CUDA_0, CUDA_0};
  m["cuFileDriverSetP2PFlags"]                              = {CUFILE_1150, CUDA_0, CUDA_0};

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_FILE_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipFileOpStatusError"]                                 = {HIP_7020, HIP_0, HIP_0};
  m["hipFileHandleRegister"]                                = {HIP_7020, HIP_0, HIP_0};
  m["hipFileHandleDeregister"]                              = {HIP_7020, HIP_0, HIP_0};
  m["hipFileBufRegister"]                                   = {HIP_7020, HIP_0, HIP_0};
  m["hipFileBufDeregister"]                                 = {HIP_7020, HIP_0, HIP_0};
  m["hipFileRead"]                                          = {HIP_7020, HIP_0, HIP_0};
  m["hipFileWrite"]                                         = {HIP_7020, HIP_0, HIP_0};
  m["hipFileDriverOpen"]                                    = {HIP_7020, HIP_0, HIP_0};
  m["hipFileDriverClose"]                                   = {HIP_7020, HIP_0, HIP_0};
  m["hipFileUseCount"]                                      = {HIP_7020, HIP_0, HIP_0};
  m["hipFileDriverGetProperties"]                           = {HIP_7020, HIP_0, HIP_0};
  m["hipFileDriverSetPollMode"]                             = {HIP_7020, HIP_0, HIP_0};
  m["hipFileDriverSetMaxDirectIOSize"]                      = {HIP_7020, HIP_0, HIP_0};
  m["hipFileDriverSetMaxCacheSize"]                         = {HIP_7020, HIP_0, HIP_0};
  m["hipFileDriverSetMaxPinnedMemSize"]                     = {HIP_7020, HIP_0, HIP_0};
  m["hipFileWriteAsync"]                                    = {HIP_7020, HIP_0, HIP_0};
  m["hipFileReadAsync"]                                     = {HIP_7020, HIP_0, HIP_0};
  m["hipFileStreamRegister"]                                = {HIP_7020, HIP_0, HIP_0};
  m["hipFileStreamDeregister"]                              = {HIP_7020, HIP_0, HIP_0};
  m["hipFileBatchIOSetUp"]                                  = {HIP_7020, HIP_0, HIP_0};
  m["hipFileBatchIOSubmit"]                                 = {HIP_7020, HIP_0, HIP_0};
  m["hipFileBatchIOGetStatus"]                              = {HIP_7020, HIP_0, HIP_0};
  m["hipFileBatchIOCancel"]                                 = {HIP_7020, HIP_0, HIP_0};
  m["hipFileBatchIODestroy"]                                = {HIP_7020, HIP_0, HIP_0};
  m["hipFileGetParameterSizeT"]                             = {HIP_7020, HIP_0, HIP_0};
  m["hipFileGetParameterBool"]                              = {HIP_7020, HIP_0, HIP_0};
  m["hipFileGetParameterString"]                            = {HIP_7020, HIP_0, HIP_0};
  m["hipFileSetParameterSizeT"]                             = {HIP_7020, HIP_0, HIP_0};
  m["hipFileSetParameterBool"]                              = {HIP_7020, HIP_0, HIP_0};
  m["hipFileSetParameterString"]                            = {HIP_7020, HIP_0, HIP_0};

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_FILE_API_SECTION_MAP = [] {
  std::map<unsigned int, llvm::StringRef> m;

  m[1]                                                      = "cuFile Types";
  m[2]                                                      = "cuFile Functions";

  return m;
}();
