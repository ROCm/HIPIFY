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
  {"cufileop_status_error",           {"hipFileOpStatusError",             "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileHandleRegister",            {"hipFileHandleRegister",            "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileHandleDeregister",          {"hipFileHandleDeregister",          "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileBufRegister",               {"hipFileBufRegister",               "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileBufDeregister",             {"hipFileBufDeregister",             "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileRead",                      {"hipFileRead",                      "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileWrite",                     {"hipFileWrite",                     "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverOpen",                {"hipFileDriverOpen",                "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverClose",               {"hipFileDriverClose",               "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverClose_v2",            {"hipFileDriverClose",               "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileUseCount",                  {"hipFileUseCount",                  "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverGetProperties",       {"hipFileDriverGetProperties",       "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverSetPollMode",         {"hipFileDriverSetPollMode",         "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverSetMaxDirectIOSize",  {"hipFileDriverSetMaxDirectIOSize",  "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverSetMaxCacheSize",     {"hipFileDriverSetMaxCacheSize",     "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileDriverSetMaxPinnedMemSize", {"hipFileDriverSetMaxPinnedMemSize", "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileWriteAsync",                {"hipFileWriteAsync",                "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileReadAsync",                 {"hipFileReadAsync",                 "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileStreamRegister",            {"hipFileStreamRegister",            "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileStreamDeregister",          {"hipFileStreamDeregister",          "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileBatchIOSetUp",              {"hipFileBatchIOSetUp",              "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileBatchIOSubmit",             {"hipFileBatchIOSubmit",             "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileBatchIOGetStatus",          {"hipFileBatchIOGetStatus",          "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileBatchIOCancel",             {"hipFileBatchIOCancel",             "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileBatchIODestroy",            {"hipFileBatchIODestroy",            "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileGetParameterSizeT",         {"hipFileGetParameterSizeT",         "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileGetParameterBool",          {"hipFileGetParameterBool",          "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileGetParameterString",        {"hipFileGetParameterString",        "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileSetParameterSizeT",         {"hipFileSetParameterSizeT",         "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileSetParameterBool",          {"hipFileSetParameterBool",          "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
  {"cuFileSetParameterString",        {"hipFileSetParameterString",        "", CONV_LIB_FUNC, API_FILE, 2, FULL}},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_FILE_FUNCTION_VER_MAP {
  {"hipFileOpStatusError",             {HIP_7000, HIP_0, HIP_0}},
  {"hipFileHandleRegister",            {HIP_7000, HIP_0, HIP_0}},
  {"hipFileHandleDeregister",          {HIP_7000, HIP_0, HIP_0}},
  {"hipFileBufRegister",               {HIP_7000, HIP_0, HIP_0}},
  {"hipFileBufDeregister",             {HIP_7000, HIP_0, HIP_0}},
  {"hipFileRead",                      {HIP_7000, HIP_0, HIP_0}},
  {"hipFileWrite",                     {HIP_7000, HIP_0, HIP_0}},
  {"hipFileDriverOpen",                {HIP_7000, HIP_0, HIP_0}},
  {"hipFileDriverClose",               {HIP_7000, HIP_0, HIP_0}},
  {"hipFileUseCount",                  {HIP_7000, HIP_0, HIP_0}},
  {"hipFileDriverGetProperties",       {HIP_7000, HIP_0, HIP_0}},
  {"hipFileDriverSetPollMode",         {HIP_7000, HIP_0, HIP_0}},
  {"hipFileDriverSetMaxDirectIOSize",  {HIP_7000, HIP_0, HIP_0}},
  {"hipFileDriverSetMaxCacheSize",     {HIP_7000, HIP_0, HIP_0}},
  {"hipFileDriverSetMaxPinnedMemSize", {HIP_7000, HIP_0, HIP_0}},
  {"hipFileWriteAsync",                {HIP_7000, HIP_0, HIP_0}},
  {"hipFileReadAsync",                 {HIP_7000, HIP_0, HIP_0}},
  {"hipFileStreamRegister",            {HIP_7000, HIP_0, HIP_0}},
  {"hipFileStreamDeregister",          {HIP_7000, HIP_0, HIP_0}},
  {"hipFileBatchIOSetUp",              {HIP_7000, HIP_0, HIP_0}},
  {"hipFileBatchIOSubmit",             {HIP_7000, HIP_0, HIP_0}},
  {"hipFileBatchIOGetStatus",          {HIP_7000, HIP_0, HIP_0}},
  {"hipFileBatchIOCancel",             {HIP_7000, HIP_0, HIP_0}},
  {"hipFileBatchIODestroy",            {HIP_7000, HIP_0, HIP_0}},
  {"hipFileGetParameterSizeT",         {HIP_7000, HIP_0, HIP_0}},
  {"hipFileGetParameterBool",          {HIP_7000, HIP_0, HIP_0}},
  {"hipFileGetParameterString",        {HIP_7000, HIP_0, HIP_0}},
  {"hipFileSetParameterSizeT",         {HIP_7000, HIP_0, HIP_0}},
  {"hipFileSetParameterBool",          {HIP_7000, HIP_0, HIP_0}},
  {"hipFileSetParameterString",        {HIP_7000, HIP_0, HIP_0}},
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

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_FILE_FUNCTION_CHANGED_VER_MAP {
  {"cufileop_status_error",           {CUDA_0}},
  {"cuFileHandleRegister",            {CUDA_0}},
  {"cuFileHandleDeregister",          {CUDA_0}},
  {"cuFileBufRegister",               {CUDA_0}},
  {"cuFileBufDeregister",             {CUDA_0}},
  {"cuFileRead",                      {CUDA_0}},
  {"cuFileWrite",                     {CUDA_0}},
  {"cuFileDriverOpen",                {CUDA_0}},
  {"cuFileDriverClose",               {CUDA_0}},
  {"cuFileDriverClose_v2",            {CUDA_0}},
  {"cuFileUseCount",                  {CUDA_0}},
  {"cuFileDriverGetProperties",       {CUDA_0}},
  {"cuFileDriverSetPollMode",         {CUDA_0}},
  {"cuFileDriverSetMaxDirectIOSize",  {CUDA_0}},
  {"cuFileDriverSetMaxCacheSize",     {CUDA_0}},
  {"cuFileDriverSetMaxPinnedMemSize", {CUDA_0}},
  {"cuFileWriteAsync",                {CUDA_0}},
  {"cuFileReadAsync",                 {CUDA_0}},
  {"cuFileStreamRegister",            {CUDA_0}},
  {"cuFileStreamDeregister",          {CUDA_0}},
  {"cuFileBatchIOSetUp",              {CUDA_0}},
  {"cuFileBatchIOSubmit",             {CUDA_0}},
  {"cuFileBatchIOGetStatus",          {CUDA_0}},
  {"cuFileBatchIOCancel",             {CUDA_0}},
  {"cuFileBatchIODestroy",            {CUDA_0}},
  {"cuFileGetParameterSizeT",         {CUDA_0}},
  {"cuFileGetParameterBool",          {CUDA_0}},
  {"cuFileGetParameterString",        {CUDA_0}},
  {"cuFileSetParameterSizeT",         {CUDA_0}},
  {"cuFileSetParameterBool",          {CUDA_0}},
  {"cuFileSetParameterString",        {CUDA_0}},
};

const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_FILE_FUNCTION_CHANGED_VER_MAP {
  {"hipFileHandleRegister",            {HIP_7000}},
  {"hipFileHandleDeregister",          {HIP_7000}},
  {"hipFileBufRegister",               {HIP_7000}},
  {"hipFileBufDeregister",             {HIP_7000}},
  {"hipFileRead",                      {HIP_7000}},
  {"hipFileWrite",                     {HIP_7000}},
  {"hipFileDriverOpen",                {HIP_7000}},
  {"hipFileDriverClose",               {HIP_7000}},
  {"hipFileUseCount",                  {HIP_7000}},
  {"hipFileDriverGetProperties",       {HIP_7000}},
  {"hipFileDriverSetPollMode",         {HIP_7000}},
  {"hipFileDriverSetMaxDirectIOSize",  {HIP_7000}},
  {"hipFileDriverSetMaxCacheSize",     {HIP_7000}},
  {"hipFileDriverSetMaxPinnedMemSize", {HIP_7000}},
  {"hipFileWriteAsync",                {HIP_7000}},
  {"hipFileReadAsync",                 {HIP_7000}},
  {"hipFileStreamRegister",            {HIP_7000}},
  {"hipFileStreamDeregister",          {HIP_7000}},
  {"hipFileBatchIOSetUp",              {HIP_7000}},
  {"hipFileBatchIOSubmit",             {HIP_7000}},
  {"hipFileBatchIOGetStatus",          {HIP_7000}},
  {"hipFileBatchIOCancel",             {HIP_7000}},
  {"hipFileBatchIODestroy",            {HIP_7000}},
  {"hipFileGetParameterSizeT",         {HIP_7000}},
  {"hipFileGetParameterBool",          {HIP_7000}},
  {"hipFileGetParameterString",        {HIP_7000}},
  {"hipFileSetParameterSizeT",         {HIP_7000}},
  {"hipFileSetParameterBool",          {HIP_7000}},
  {"hipFileSetParameterString",        {HIP_7000}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_FILE_API_SECTION_MAP {
  {1, "cuFile Types"},
  {2, "cuFile Functions"},
};
