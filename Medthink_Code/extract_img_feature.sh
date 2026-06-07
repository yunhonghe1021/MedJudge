#!/bin/bash

python extract_img_feature.py --device cuda:0 --image_dir data/R-RAD/images/ --output_dir  data/R-RAD/
python extract_img_feature.py --device cuda:0 --image_dir data/R-SLAKE/img/ --output_dir  data/R-SLAKE/

