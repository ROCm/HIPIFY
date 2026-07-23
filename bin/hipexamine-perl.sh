#!/bin/bash

#usage : hipexamine-perl.sh DIRNAME [hipify-perl options]

# Generate HIP stats (LOC, CUDA->API conversions, missing functionality) for all the code files
# in the specified directory.


SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if [ "$#" -gt 0 ]; then
  SEARCH_DIR=$1
  shift
else
  SEARCH_DIR=.
fi

mapfile -d '' -t files < <("$SCRIPT_DIR/findcode.sh" "$SEARCH_DIR")
if [ "${#files[@]}" -eq 0 ]; then
  exit 0
fi

"$SCRIPT_DIR/hipify-perl" -no-output -print-stats "$@" "${files[@]}"
