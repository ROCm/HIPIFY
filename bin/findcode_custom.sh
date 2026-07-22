#!/bin/bash

# Outputs matched .cu files (excluding .cuh), NUL-delimited.

if [ "$#" -eq 0 ]; then
  exit 0
fi

find "$@" \( \
  \( -name '*.cu' -a -not -name '*.cuh' \) -o \
  \( -name '*.CU' -a -not -name '*.CUH' \) \
\) -print0
