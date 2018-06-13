#!/bin/bash

wget -O model_data/yolo.weights http://pjreddie.com/media/files/yolo.weights
./yad2k_utils.py model_data/yolo.cfg model_data/yolo.weights model_data/yolo.h5