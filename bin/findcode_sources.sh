#!/bin/bash

# Outputs matched source files (excluding headers), NUL-delimited.

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ -z "$1" ]; }; then
  set -- .
fi

find "$@" -name '*.cu' -a -not -name '*.cuh' -print0
find "$@" -name '*.CU' -a -not -name '*.CUH' -print0
find "$@" \( -name '*.cpp' -o -name '*.cxx' -o -name '*.c' -o -name '*.cc' \) -print0
find "$@" \( -name '*.CPP' -o -name '*.CXX' -o -name '*.C' -o -name '*.CC' \) -print0
