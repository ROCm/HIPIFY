#!/bin/bash

#usage : hipexamine-perl.sh DIRNAME [hipify-perl options]

# Generate HIP stats (LOC, CUDA->API conversions, missing functionality) for all the code files
# in the specified directory.


SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PRIV_SCRIPT_DIR="$SCRIPT_DIR/../libexec/hipify"
SEARCH_DIR=$1
shift
$SCRIPT_DIR/hipify-perl -no-output -print-stats "$@" `$PRIV_SCRIPT_DIR/findcode.sh $SEARCH_DIR`
