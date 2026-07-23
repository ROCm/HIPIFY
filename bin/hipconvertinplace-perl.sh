#!/bin/bash

#usage : hipconvertinplace-perl.sh DIRNAME [-filter=all|headers|sources|custom] [hipify-perl options]

#hipify "inplace" all code files in specified directory.
# This can be quite handy when dealing with an existing CUDA code base since the script
# preserves the existing directory structure.

#  For each code file, this script will:
#   - If ".prehip file does not exist, copy the original code to a new file with extension ".prehip". Then hipify the code file.
#   - If ".prehip" file exists, this is used as input to hipify.
# (this is useful for testing improvements to the hipify-perl toolset).

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_NAME=findcode.sh
SEARCH_DIR=$1
if [ "$2" = "-filter=all" ]
then
shift
elif [ "$2" = "-filter=headers" ]
then
SCRIPT_NAME=findcode_headers.sh
shift
elif [ "$2" = "-filter=sources" ]
then
SCRIPT_NAME=findcode_sources.sh
shift
elif [ "$2" = "-filter=custom" ]
then
SCRIPT_NAME=findcode_custom.sh
shift
fi
shift

SEARCH_DIR=${SEARCH_DIR:-.}

mapfile -d '' -t files < <("$SCRIPT_DIR/$SCRIPT_NAME" "$SEARCH_DIR")
if [ "${#files[@]}" -eq 0 ]; then
  exit 0
fi

"$SCRIPT_DIR/hipify-perl" -inplace -print-stats "$@" "${files[@]}"
