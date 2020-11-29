#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

echo ""
echo ">>  Installing a third-party 2D keypoint detector"
sh scripts/install_pose2d.sh

echo ""
echo ">>  Installing a third-party hand detector"
sh scripts/install_hand_detectors.sh

echo ""
if [ ! -d "sample_data" ] 
then
    echo "Downloading sample videos"
    wget https://dl.fbaipublicfiles.com/eft/sampledata_frank.tar && tar -xvf sampledata_frank.tar && rm sampledata_frank.tar && mv sampledata sample_data
else
    echo "There exists sample_data already"
fi