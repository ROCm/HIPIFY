#!/bin/bash

# Outputs matched header files, NUL-delimited.

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ -z "$1" ]; }; then
  set -- .
fi

find "$@" \( -name '*.cuh' -o -name '*.CUH' \) -print0
find "$@" \( -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.inc' -o -name '*.inl' -o -name '*.hxx' -o -name '*.hdl' \) -print0
find "$@" \( -name '*.H' -o -name '*.HPP' -o -name '*.HH' -o -name '*.INC' -o -name '*.INL' -o -name '*.HXX' -o -name '*.HDL' \) -print0
