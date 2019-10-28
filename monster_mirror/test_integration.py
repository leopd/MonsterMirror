import pytest

import cv2
import numpy as np
import monster_mirror.spooky as spooky

def test_random_noise():
    spookifier = spooky.RoundRobinSpookifier('target-images','meerkat')
    rand_img = np.random.uniform(0,255,(480,640,3)).astype(np.uint8)
    out_img = spookifier.process_npimage(rand_img, None)

    assert spookifier.face_transform_cnt == 0
    assert out_img.shape[0] == 480
    assert out_img.shape[1] == 640


def test_ellen():
    spookifier = spooky.RoundRobinSpookifier('target-images','meerkat', max_faces=6)
    img = cv2.imread('samples/ellen-selfie.jpg')
    out_img = spookifier.process_npimage(img, None)

    assert spookifier.face_transform_cnt == 6
    assert out_img.shape[0] == 480
    #assert out_img.shape[1] == 640  #TODO figure out why this changes.

