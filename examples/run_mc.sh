#!/bin/bash

rm -rf FitsMC

peakfit  \
    -s pseudo3d.ft2 \
    -l pseudo3d.list \
    -z b1_offsets.txt \
    --mc 9.7ppm 10.5ppm 113ppm 128ppm 50 \
    -o FitsMC

plot_cest -f FitsMC/*N-H.out --ref 0
