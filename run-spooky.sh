#!/bin/bash

python monster_mirror/localcam.py \
    --extra_detail 1 \
    --max_faces 7 \
    --cycle_delay 7 \
    -t meerkat,pomeranian,tiger \
    --color_map 0.5,1,0.7 \
    --scale_embedding 0.15 \
    --grow_facebox 1.0 \
    --noise_mag 3 \
    --noise_speed 2.5 \
    --noise_drift 0.3 \
    $@

