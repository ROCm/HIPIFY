#!/bin/bash

SEARCH_DIRS=$@

find $SEARCH_DIRS -name '*.cu' -and -not -name '*.cuh'
find $SEARCH_DIRS -name '*.CU' -and -not -name '*.CUH'
