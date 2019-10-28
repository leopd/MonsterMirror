#!/bin/bash

python monster_mirror/localcam.py \
    --extra_detail 1 \
    --max_faces 7 \
    --cycle_delay 15 \
    -t meerkat,pomeranian,tiger \
    --color_map 1,1,1 \
    --scale_embedding 1.0 \
    --grow_facebox 1.0 \
    --noise_mag 5 \
    --noise_speed 2.5 \
    --noise_drift 0.3 \
    --max_alpha 0.85 \
    $@

