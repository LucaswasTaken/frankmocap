#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

echo ""
echo ">>  Installing a third-party 2D keypoint detector"
sh scripts/install_pose2d.sh

echo ""
echo ">>  Installing a third-party hand detector"
sh scripts/install_hand_detectors.sh

