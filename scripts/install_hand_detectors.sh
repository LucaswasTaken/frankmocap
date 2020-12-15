#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

mkdir -p detectors
cd detectors

pip install gdown

# Install 100-DOH hand-object detectors
git clone https://github.com/ddshan/hand_object_detector

# compile
cd hand_object_detector/lib
python setup.py build develop
cd ../../

# Install 100-DOH hand-only detectors
git clone https://github.com/ddshan/hand_detector.d2.git

mv hand_detector.d2 hand_only_detector
