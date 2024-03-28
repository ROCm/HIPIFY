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

#include <map>
#include <vector>
#include <string>

namespace hipify {

  enum CastTypes {
    e_HIP_SYMBOL,
    e_reinterpret_cast,
    e_reinterpret_cast_size_t,
    e_int32_t,
    e_int64_t,
    e_remove_argument,
    e_add_const_argument,
    e_add_var_argument,
    e_move_argument,
    e_replace_argument_with_const,
  };

  enum OverloadTypes {
    ot_arguments_number,
  };

  enum CastWarning {
    cw_None,
    cw_DataLoss,
  };

  enum OverloadWarning {
    ow_None,
  };

  struct CastInfo {
    CastTypes castType;
    CastWarning castWarn;
    std::string constValToAddOrReplace = "";
    unsigned moveOrCopyTo = 0;
    unsigned numberToMoveOrCopy = 1;
  };

  typedef std::map<unsigned, CastInfo> ArgCastMap;

  struct ArgCastStruct {
    ArgCastMap castMap;
    bool isToRoc = false;
    bool isToMIOpen = false;
  };

  struct OverloadInfo {
    hipCounter counter;
    OverloadTypes overloadType;
    OverloadWarning overloadWarn;
  };

  typedef std::map<unsigned, OverloadInfo> OverloadMap;

  struct FuncOverloadsStruct {
    OverloadMap overloadMap;
    bool isToRoc = false;
    bool isToMIOpen = false;
  };
}

extern std::string getCastType(hipify::CastTypes c);
extern std::map<std::string, std::vector<hipify::ArgCastStruct>> FuncArgCasts;

extern std::map<std::string, hipify::FuncOverloadsStruct> FuncOverloads;

extern std::map<std::string, std::string> TypeOverloads;

namespace perl {

  bool generate(bool Generate = true);
}

namespace python {

  bool generate(bool Generate = true);
}

namespace doc {

  bool generate(bool GenerateMD = true, bool GenerateCSV = true);
}
