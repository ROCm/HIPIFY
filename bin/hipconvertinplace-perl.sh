#!/bin/bash

#usage : hipconvertinplace-perl.sh DIRNAME [hipify-perl options]

#hipify "inplace" all code files in specified directory.
# This can be quite handy when dealing with an existing CUDA code base since the script
# preserves the existing directory structure.

#  For each code file, this script will:
#   - If ".prehip file does not exist, copy the original code to a new file with extension ".prehip". Then hipify the code file.
#   - If ".prehip" file exists, this is used as input to hipify.
# (this is useful for testing improvements to the hipify-perl toolset).

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PRIV_SCRIPT_DIR="$SCRIPT_DIR/../libexec/hipify"
SEARCH_DIR=$1
shift
$SCRIPT_DIR/hipify-perl -inplace -print-stats "$@" `$PRIV_SCRIPT_DIR/findcode.sh $SEARCH_DIR`
