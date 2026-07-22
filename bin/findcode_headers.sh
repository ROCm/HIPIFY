#!/bin/bash

# Outputs matched header files, NUL-delimited.

if [ "$#" -eq 0 ]; then
  exit 0
fi

find "$@" \( \
  -name '*.cuh' -o -name '*.CUH' -o \
  -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.inc' -o -name '*.inl' -o -name '*.hxx' -o -name '*.hdl' -o \
  -name '*.H' -o -name '*.HPP' -o -name '*.HH' -o -name '*.INC' -o -name '*.INL' -o -name '*.HXX' -o -name '*.HDL' \
\) -print0
