#!/bin/bash

# Outputs matched source files (excluding headers), NUL-delimited.

if [ "$#" -eq 0 ]; then
  exit 0
fi

find "$@" \( \
  \( -name '*.cu' -a -not -name '*.cuh' \) -o \
  \( -name '*.CU' -a -not -name '*.CUH' \) -o \
  -name '*.cpp' -o -name '*.cxx' -o -name '*.c' -o -name '*.cc' -o \
  -name '*.CPP' -o -name '*.CXX' -o -name '*.C' -o -name '*.CC' \
\) -print0
