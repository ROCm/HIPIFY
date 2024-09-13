#!/bin/bash

SEARCH_DIRS=$@

find $SEARCH_DIRS -name '*.cu' -and -not -name '*.cuh'
find $SEARCH_DIRS -name '*.CU' -and -not -name '*.CUH'
find $SEARCH_DIRS -name '*.cpp' -o -name '*.cxx' -o -name '*.c' -o -name '*.cc'
find $SEARCH_DIRS -name '*.CPP' -o -name '*.CXX' -o -name '*.C' -o -name '*.CC'
