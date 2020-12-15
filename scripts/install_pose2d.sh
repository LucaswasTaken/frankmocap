#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

mkdir -p detectors
cd detectors

git clone https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git
mv lightweight-human-pose-estimation.pytorch body_pose_estimator
