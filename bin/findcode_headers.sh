#!/bin/bash

SEARCH_DIRS=$@

find $SEARCH_DIRS -name '*.cuh' -o -name '*.CUH'
find $SEARCH_DIRS -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.inc' -o -name '*.inl' -o -name '*.hxx' -o -name '*.hdl'
find $SEARCH_DIRS -name '*.H' -o -name '*.HPP' -o -name '*.HH' -o -name '*.INC' -o -name '*.INL' -o -name '*.HXX' -o -name '*.HDL'
