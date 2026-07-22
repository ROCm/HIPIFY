#!/bin/bash

# Outputs files that do not match known CUDA/HIP source/header extensions, NUL-delimited.

SEARCH_DIR=$1

if [ -z "$SEARCH_DIR" ]; then
  exit 0
fi

find "$SEARCH_DIR" \( \
  -not -name '*.cu' -a \
  -not -name '*.cpp' -a \
  -not -name '*.cxx' -a \
  -not -name '*.c' -a \
  -not -name '*.cc' -a \
  -not -name '*.cuh' -a \
  -not -name '*.h' -a \
  -not -name '*.hpp' -a \
  -not -name '*.inc' -a \
  -not -name '*.inl' -a \
  -not -name '*.hxx' -a \
  -not -name '*.hdl' \
\) -print0
