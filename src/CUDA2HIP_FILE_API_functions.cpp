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

const std::map<llvm::StringRef, hipCounter> CUDA_FILE_FUNCTION_MAP {
  {"cufileop_status_error",           {"hipFileOpStatusError",             "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileHandleRegister",            {"hipFileHandleRegister",            "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileHandleDeregister",          {"hipFileHandleDeregister",          "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileBufRegister",               {"hipFileBufRegister",               "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileBufDeregister",             {"hipFileBufDeregister",             "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileRead",                      {"hipFileRead",                      "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileWrite",                     {"hipFileWrite",                     "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverOpen",                {"hipFileDriverOpen",                "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverClose",               {"hipFileDriverClose",               "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverClose_v2",            {"hipFileDriverClose",               "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileUseCount",                  {"hipFileUseCount",                  "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverGetProperties",       {"hipFileDriverGetProperties",       "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverSetPollMode",         {"hipFileDriverSetPollMode",         "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverSetMaxDirectIOSize",  {"hipFileDriverSetMaxDirectIOSize",  "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverSetMaxCacheSize",     {"hipFileDriverSetMaxCacheSize",     "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileDriverSetMaxPinnedMemSize", {"hipFileDriverSetMaxPinnedMemSize", "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileWriteAsync",                {"hipFileWriteAsync",                "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileReadAsync",                 {"hipFileReadAsync",                 "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileStreamRegister",            {"hipFileStreamRegister",            "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileStreamDeregister",          {"hipFileStreamDeregister",          "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileBatchIOSetUp",              {"hipFileBatchIOSetUp",              "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileBatchIOSubmit",             {"hipFileBatchIOSubmit",             "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileBatchIOGetStatus",          {"hipFileBatchIOGetStatus",          "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileBatchIOCancel",             {"hipFileBatchIOCancel",             "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileBatchIODestroy",            {"hipFileBatchIODestroy",            "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileGetParameterSizeT",         {"hipFileGetParameterSizeT",         "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileGetParameterBool",          {"hipFileGetParameterBool",          "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileGetParameterString",        {"hipFileGetParameterString",        "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileSetParameterSizeT",         {"hipFileSetParameterSizeT",         "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileSetParameterBool",          {"hipFileSetParameterBool",          "", CONV_LIB_FUNC, API_FILE, 2}},
  {"cuFileSetParameterString",        {"hipFileSetParameterString",        "", CONV_LIB_FUNC, API_FILE, 2}},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_FILE_FUNCTION_VER_MAP {
  {"hipFileOpStatusError",             {HIP_7020, HIP_0, HIP_0}},
  {"hipFileHandleRegister",            {HIP_7020, HIP_0, HIP_0}},
  {"hipFileHandleDeregister",          {HIP_7020, HIP_0, HIP_0}},
  {"hipFileBufRegister",               {HIP_7020, HIP_0, HIP_0}},
  {"hipFileBufDeregister",             {HIP_7020, HIP_0, HIP_0}},
  {"hipFileRead",                      {HIP_7020, HIP_0, HIP_0}},
  {"hipFileWrite",                     {HIP_7020, HIP_0, HIP_0}},
  {"hipFileDriverOpen",                {HIP_7020, HIP_0, HIP_0}},
  {"hipFileDriverClose",               {HIP_7020, HIP_0, HIP_0}},
  {"hipFileUseCount",                  {HIP_7020, HIP_0, HIP_0}},
  {"hipFileDriverGetProperties",       {HIP_7020, HIP_0, HIP_0}},
  {"hipFileDriverSetPollMode",         {HIP_7020, HIP_0, HIP_0}},
  {"hipFileDriverSetMaxDirectIOSize",  {HIP_7020, HIP_0, HIP_0}},
  {"hipFileDriverSetMaxCacheSize",     {HIP_7020, HIP_0, HIP_0}},
  {"hipFileDriverSetMaxPinnedMemSize", {HIP_7020, HIP_0, HIP_0}},
  {"hipFileWriteAsync",                {HIP_7020, HIP_0, HIP_0}},
  {"hipFileReadAsync",                 {HIP_7020, HIP_0, HIP_0}},
  {"hipFileStreamRegister",            {HIP_7020, HIP_0, HIP_0}},
  {"hipFileStreamDeregister",          {HIP_7020, HIP_0, HIP_0}},
  {"hipFileBatchIOSetUp",              {HIP_7020, HIP_0, HIP_0}},
  {"hipFileBatchIOSubmit",             {HIP_7020, HIP_0, HIP_0}},
  {"hipFileBatchIOGetStatus",          {HIP_7020, HIP_0, HIP_0}},
  {"hipFileBatchIOCancel",             {HIP_7020, HIP_0, HIP_0}},
  {"hipFileBatchIODestroy",            {HIP_7020, HIP_0, HIP_0}},
  {"hipFileGetParameterSizeT",         {HIP_7020, HIP_0, HIP_0}},
  {"hipFileGetParameterBool",          {HIP_7020, HIP_0, HIP_0}},
  {"hipFileGetParameterString",        {HIP_7020, HIP_0, HIP_0}},
  {"hipFileSetParameterSizeT",         {HIP_7020, HIP_0, HIP_0}},
  {"hipFileSetParameterBool",          {HIP_7020, HIP_0, HIP_0}},
  {"hipFileSetParameterString",        {HIP_7020, HIP_0, HIP_0}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_FILE_FUNCTION_VER_MAP {
  {"cufileop_status_error",           {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileHandleRegister",            {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileHandleDeregister",          {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileBufRegister",               {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileBufDeregister",             {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileRead",                      {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileWrite",                     {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverOpen",                {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverClose",               {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverClose_v2",            {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileUseCount",                  {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverGetProperties",       {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverSetPollMode",         {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverSetMaxDirectIOSize",  {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverSetMaxCacheSize",     {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileDriverSetMaxPinnedMemSize", {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileWriteAsync",                {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileReadAsync",                 {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileStreamRegister",            {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileStreamDeregister",          {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileBatchIOSetUp",              {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileBatchIOSubmit",             {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileBatchIOGetStatus",          {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileBatchIOCancel",             {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileBatchIODestroy",            {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileGetParameterSizeT",         {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileGetParameterBool",          {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileGetParameterString",        {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileSetParameterSizeT",         {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileSetParameterBool",          {CUDA_129, CUDA_0, CUDA_0}},
  {"cuFileSetParameterString",        {CUDA_129, CUDA_0, CUDA_0}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_FILE_API_SECTION_MAP {
  {1, "cuFile Types"},
  {2, "cuFile Functions"},
};
