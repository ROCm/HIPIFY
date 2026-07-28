#!/bin/bash

#usage : hipexamine.sh DIRNAME [hipify options] [--] [clang options]

# Generate CUDA->HIP conversion statistics for all the code files in the specified directory.

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
BIN_DIR="$SCRIPT_DIR/../../bin"
if [ "$#" -gt 0 ]; then
  SEARCH_DIR=$1
  shift
else
  SEARCH_DIR=.
fi

hipify_args=()
clang_args=()
parsing_clang=0
for arg in "$@"; do
  if [ "$parsing_clang" -eq 1 ]; then
    clang_args+=("$arg")
  elif [ "$arg" = "--" ]; then
    parsing_clang=1
  else
    hipify_args+=("$arg")
  fi
done

mapfile -d '' -t files < <("$SCRIPT_DIR/findcode.sh" "$SEARCH_DIR")
if [ "${#files[@]}" -eq 0 ]; then
  exit 0
fi

"$BIN_DIR/hipify-clang" -examine "${hipify_args[@]}" "${files[@]}" -- -x cuda "${clang_args[@]}"
