#!/bin/bash

# Outputs matched CUDA/HIP source and header files, NUL-delimited.

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ -z "$1" ]; }; then
  set -- .
fi

find "$@" \( -name '*.cu' -o -name '*.CU' \) -print0
find "$@" \( -name '*.cpp' -o -name '*.cxx' -o -name '*.c' -o -name '*.cc' \) -print0
find "$@" \( -name '*.CPP' -o -name '*.CXX' -o -name '*.C' -o -name '*.CC' \) -print0
find "$@" \( -name '*.cuh' -o -name '*.CUH' \) -print0
find "$@" \( -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.inc' -o -name '*.inl' -o -name '*.hxx' -o -name '*.hdl' \) -print0
find "$@" \( -name '*.H' -o -name '*.HPP' -o -name '*.HH' -o -name '*.INC' -o -name '*.INL' -o -name '*.HXX' -o -name '*.HDL' \) -print0
