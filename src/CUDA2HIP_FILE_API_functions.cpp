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
  {"cuFileGetVersion",                {"hipFileGetVersion",                "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileGetParameterMinMaxValue",   {"hipFileGetParameterMinMaxValue",   "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileSetStatsLevel",             {"hipFileSetStatsLevel",             "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileGetStatsLevel",             {"hipFileGetStatsLevel",             "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileStatsStart",                {"hipFileStatsStart",                "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileStatsStop",                 {"hipFileStatsStop",                 "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileStatsReset",                {"hipFileStatsReset",                "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileGetStatsL1",                {"hipFileGetStatsL1",                "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileGetStatsL2",                {"hipFileGetStatsL2",                "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileGetStatsL3",                {"hipFileGetStatsL3",                "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileGetBARSizeInKB",            {"hipFileGetBARSizeInKB",            "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileSetParameterPosixPoolSlabArray",  {"hipFileSetParameterPosixPoolSlabArray",  "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileGetParameterPosixPoolSlabArray",  {"hipFileGetParameterPosixPoolSlabArray",  "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileDriverGetP2PFlags",         {"hipFileDriverGetP2PFlags",         "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
  {"cuFileDriverSetP2PFlags",         {"hipFileDriverSetP2PFlags",         "", CONV_LIB_FUNC, API_FILE, 2, UNSUPPORTED}},
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
  {"cufileop_status_error",           {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileHandleRegister",            {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileHandleDeregister",          {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileBufRegister",               {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileBufDeregister",             {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileRead",                      {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileWrite",                     {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileDriverOpen",                {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileDriverClose",               {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileDriverClose_v2",            {CUFILE_1040, CUDA_0, CUDA_0}},
  {"cuFileUseCount",                  {CUFILE_1040, CUDA_0, CUDA_0}},
  {"cuFileDriverGetProperties",       {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileDriverSetPollMode",         {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileDriverSetMaxDirectIOSize",  {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileDriverSetMaxCacheSize",     {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileDriverSetMaxPinnedMemSize", {CUFILE_1000, CUDA_0, CUDA_0}},
  {"cuFileWriteAsync",                {CUFILE_1070, CUDA_0, CUDA_0}},
  {"cuFileReadAsync",                 {CUFILE_1070, CUDA_0, CUDA_0}},
  {"cuFileStreamRegister",            {CUFILE_1070, CUDA_0, CUDA_0}},
  {"cuFileStreamDeregister",          {CUFILE_1070, CUDA_0, CUDA_0}},
  {"cuFileBatchIOSetUp",              {CUFILE_1020, CUDA_0, CUDA_0}},
  {"cuFileBatchIOSubmit",             {CUFILE_1020, CUDA_0, CUDA_0}},
  {"cuFileBatchIOGetStatus",          {CUFILE_1020, CUDA_0, CUDA_0}},
  {"cuFileBatchIOCancel",             {CUFILE_1020, CUDA_0, CUDA_0}},
  {"cuFileBatchIODestroy",            {CUFILE_1020, CUDA_0, CUDA_0}},
  {"cuFileGetParameterSizeT",         {CUFILE_1140, CUDA_0, CUDA_0}},
  {"cuFileGetParameterBool",          {CUFILE_1140, CUDA_0, CUDA_0}},
  {"cuFileGetParameterString",        {CUFILE_1140, CUDA_0, CUDA_0}},
  {"cuFileSetParameterSizeT",         {CUFILE_1140, CUDA_0, CUDA_0}},
  {"cuFileSetParameterBool",          {CUFILE_1140, CUDA_0, CUDA_0}},
  {"cuFileSetParameterString",        {CUFILE_1140, CUDA_0, CUDA_0}},
  {"cuFileGetVersion",                {CUFILE_1080, CUDA_0, CUDA_0}},
  {"cuFileGetParameterMinMaxValue",   {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileSetStatsLevel",             {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileGetStatsLevel",             {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileStatsStart",                {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileStatsStop",                 {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileStatsReset",                {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileGetStatsL1",                {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileGetStatsL2",                {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileGetStatsL3",                {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileGetBARSizeInKB",            {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileSetParameterPosixPoolSlabArray",  {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileGetParameterPosixPoolSlabArray",  {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileDriverGetP2PFlags",         {CUFILE_1150, CUDA_0, CUDA_0}},
  {"cuFileDriverSetP2PFlags",         {CUFILE_1150, CUDA_0, CUDA_0}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_FILE_API_SECTION_MAP {
  {1, "cuFile Types"},
  {2, "cuFile Functions"},
};
