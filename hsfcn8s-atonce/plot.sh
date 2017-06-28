#!/bin/bash

data=`\
cat solve.log | \
grep ", loss = " | \
awk '{ print $6,$13 }'`

echo "$data" | gnuplot -p -e "plot '-' with lines title 'loss' "
