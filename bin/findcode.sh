#!/bin/bash

# Outputs matched CUDA/HIP source and header files, NUL-delimited.

if [ "$#" -eq 0 ]; then
  exit 0
fi

find "$@" \( \
  -name '*.cu' -o -name '*.CU' -o \
  -name '*.cpp' -o -name '*.cxx' -o -name '*.c' -o -name '*.cc' -o \
  -name '*.CPP' -o -name '*.CXX' -o -name '*.C' -o -name '*.CC' -o \
  -name '*.cuh' -o -name '*.CUH' -o \
  -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.inc' -o -name '*.inl' -o -name '*.hxx' -o -name '*.hdl' -o \
  -name '*.H' -o -name '*.HPP' -o -name '*.HH' -o -name '*.INC' -o -name '*.INL' -o -name '*.HXX' -o -name '*.HDL' \
\) -print0
