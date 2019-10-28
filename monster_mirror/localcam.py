"""
Modified BSD License for Live Performance Only

Copyright (c) 2019, Leo Dirac
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. This software and any derivates must be used for live performances only.
   Any media content produced by this software may not be stored nor archived 
   electronically. Any transmission of said media must be limited to immediate 
   ephemeral display within clear view of any people whose likeness is being 
   recorded, captured, or modified. Exception to this clause is only allowed 
   with express written consent of every individual whose likeness is being 
   recorded, captured, or modified, as well as any individual being portrayed,
   represented, or impersonated.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__doc__ = '''MonsterMirror

For Live Performances Only.  See LICENSE for details.
'''

import argparse
import cv2
import os
import numpy as np
import sys
import time
from timebudget import timebudget
import traceback

# Add the parent directory to the python path to get funit, sfd_pytorch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import spooky

def flip_img(img:np.ndarray) -> np.ndarray:
    """Mirror image left-right
    """
    return img[:,::-1,:]

def local_spooky(no_full_screen:bool, **kwargs):
    if not no_full_screen:
        for _ in range(10):
            print("Will run full screen.  Press SPACEBAR key to exit...\n\n")
            time.sleep(0.05)
    spookifier = spooky.RoundRobinSpookifier(**kwargs)

    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FPS, 15)
        start_time = time.time()
        frame_cnt = 0
        show_perf = True
        if no_full_screen:
            display = cv2.namedWindow("Spooky")
        else:
            display = cv2.namedWindow("Spooky", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Spooky", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            frame_cnt += 1
            fps = frame_cnt / (time.time() - start_time)
            if (frame_cnt % 3 == 0) and (show_perf):
                print("\033[H\033[J")
                print(f"------------ At frame {frame_cnt} average speed is {fps:.2f}fps")
                timebudget.report('process_npimage')
            status_code, img = camera.read()
            img = spookifier.process_npimage(img, None)
            flipped = flip_img(img)
            cv2.imshow('Spooky', flipped)
            key = cv2.waitKey(1)
            if key == ord('e') or key == ord('E'):  # E = embedding capture
                e = spookifier.get_target_embedding()
                with open("embeddings.txt","at") as f:
                    f.write(str(e))
                    f.write("\n")
                    print("Recorded embedding")
            elif key == ord('d') or key == ord('D'):  # D = debug
                current_time = time.time()
                import pdb; pdb.set_trace()
            elif key == ord('p') or key == ord('P'):  # P = perf data
                show_perf = not show_perf
            elif key == 32 or key == ord('Q') or key == ord('q'):
                print(f"Quitting")
                return
            elif key >= 0:
                print(f"Key #{key} pressed.  No action for this key.")
    finally:
        del(camera)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target_image_base',
        type=str,
        default='target-images',
        help='Folder where target image subdirs are located')
    parser.add_argument('-t', '--target_classes',
        type=str,
        default='tiger,meerkat',
        help='List of target class names. Must be folders under target_image_base')
    parser.add_argument('-nfs', '--no_full_screen',
        action='store_true',
        default=False,
        help='Run windowed instead of full screen.')
    parser.add_argument('-gf', '--grow_facebox',
        type=float,
        default=0.3,
        help='factor to increase size of facebox by')
    parser.add_argument('-mf', '--max_faces',
        type=int,
        default=4,
        help='maximum number of faces to process')
    parser.add_argument('-ed', '--extra_detail',
        type=int,
        default=1,
        help='number of extra CNN passes to refine detail')
    parser.add_argument('-cyc', '--cycle_delay',
        type=float,
        default=5.0,
        help='Number of seconds between switching classes')
    parser.add_argument('-ns', '--noise_speed',
        type=float,
        default=0.7,
        help='Number of noise cycles per class')
    parser.add_argument('-nm', '--noise_mag',
        type=float,
        default=3,
        help='How large the noise vector can get')
    parser.add_argument('-nd', '--noise_drift',
        type=float,
        default=0.1,
        help='How quickly the noise vector should change')
    parser.add_argument('-cm', '--color_map',
        type=str,
        default='0.7,1,0.5',
        help='Color map as RGB multipliers. e.g. 1,1,1 is true-color')
    parser.add_argument('-se', '--scale_embedding',
        type=float,
        default=0.2,
        help='Scale embedding. 1.0 for full animal. Lower for freaky')
    parser.add_argument('-ma', '--max_alpha',
        type=float,
        default=0.7,
        help='Max alpha to blend generated images back in with')
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    kwargs = dict(opts.__dict__)
    local_spooky(**kwargs)
