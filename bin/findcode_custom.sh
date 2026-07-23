#!/bin/bash

# Outputs matched .cu files (excluding .cuh), NUL-delimited.

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ -z "$1" ]; }; then
  set -- .
fi

find "$@" -name '*.cu' -a -not -name '*.cuh' -print0
find "$@" -name '*.CU' -a -not -name '*.CUH' -print0
