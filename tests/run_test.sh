#!/usr/bin/env bash

set -o errexit

# Run a single LIT test file in a magical way that preserves colour output, to work around
# a known flaw in lit.

# Capture lit substitutions
HIPIFY=$1
IN_FILE=$2
TMP_FILE=$3
CUDA_ROOT=$4
CLANG_RES=$5
NUM=$6
HIPIFY_OPTS=""

if [ $NUM = "1" ]
then
HIPIFY_OPTS=$7
shift 7
elif [ $NUM = "2" ]
then
HIPIFY_OPTS="$7 $8"
shift 8
elif [ $NUM = "3" ]
then
HIPIFY_OPTS="$7 $8 $9"
shift 9
elif [ $NUM = "4" ]
then
HIPIFY_OPTS="$7 $8 $9 ${10}"
shift 10
elif [ $NUM = "5" ]
then
HIPIFY_OPTS="$7 $8 $9 ${10} ${11}"
shift 11
fi

test_dir=${IN_FILE%/*}

compile_commands=compile_commands.json
json_out=${test_dir}/${compile_commands}
json_in=${json_out}.in

# Remaining args are the ones to forward to clang proper.

if [ -e $json_in ]
then
cp $json_in $json_out
sed -i -e "s|<test dir>|${test_dir}|g; s|<CUDA dir>|${CUDA_ROOT}|g" $json_out
$HIPIFY -o=$TMP_FILE $IN_FILE $CUDA_ROOT --clang-resource-directory=$CLANG_RES $HIPIFY_OPTS -p=$test_dir $HIPIFY_OPTS && cat $TMP_FILE | sed -Ee 's|//.+|// |g' | FileCheck $IN_FILE
else
$HIPIFY -o=$TMP_FILE $IN_FILE $CUDA_ROOT --clang-resource-directory=$CLANG_RES $HIPIFY_OPTS -- $@ && cat $TMP_FILE | sed -Ee 's|//.+|// |g' | FileCheck $IN_FILE
fi
