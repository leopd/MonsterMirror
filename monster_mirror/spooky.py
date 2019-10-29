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
import argparse
import cv2
import functools
import math
import numpy as np
import os
from PIL import Image
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision
from typing import Tuple

from nntools.maybe_cuda import mbcuda
from funit.utils import get_config
from funit.trainer import Trainer
from sfd_pytorch import S3fd_Model, detect_faces
from timebudget import timebudget


class Spookifier():

    def __init__(self, 
                    config_file:str='funit/configs/funit_animals.yaml',
                    face_finder_model:str='pretrained-models/s3fd_convert.pth',
                    funit_model:str='pretrained-models/animal149_gen.pt',
                    target_image_folder:str='target-images/meerkat',
                    grow_facebox:float=0.2,
                    cycle_delay:float=5.0,
                    extra_detail:int=2,
                    min_face_size:int=20,
                    max_faces:int=5,
                    color_map:str='1,1,1',
                    scale_embedding:float=1.0,
                    max_alpha:float=0.7,
                ):
        self.face_transform_cnt = 0
        self.grow_facebox = grow_facebox
        self.extra_detail = extra_detail
        self.cycle_delay = cycle_delay
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self.set_color(*[float(n) for n in color_map.split(',')])
        self.scale_embedding = scale_embedding
        self.max_alpha = max_alpha

        print("Loading face detector...")
        self.face_detect_model = S3fd_Model()
        self.face_detect_model.load_state_dict(torch.load(face_finder_model))
        mbcuda(self.face_detect_model)
        self.face_detect_model.eval()

        print("Loading trainer...")
        config = get_config(config_file)
        self.trainer = Trainer(config)
        mbcuda(self.trainer)
        self.trainer.load_ckpt(funit_model)
        self.trainer.eval()

        print("Loading transfomer...")
        transform_list = [torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list = [torchvision.transforms.Resize((128, 128))] + transform_list
        self.transform = torchvision.transforms.Compose(transform_list)

        self.target_embedding = self.target_embedding_from_images(target_image_folder)

    def set_color(self, R:float, G:float, B:float):
        self.colorshift = mbcuda(torch.Tensor([[[R,G,B]]]))

    def target_embedding_from_images(self, target_image_folder:str) -> torch.Tensor:
        images = os.listdir(target_image_folder)
        print(f"Found {len(images)} target images in {target_image_folder}")
        new_class_code = None
        for i, f in enumerate(images):
            if f.startswith('.') or f=="LICENSE":
                continue  # .DS_Store or ._whatever
            fn = os.path.join(target_image_folder, f)
            img = Image.open(fn).convert('RGB')
            img_tensor = mbcuda(self.transform(img).unsqueeze(0))
            with torch.no_grad():
                class_code = self.trainer.model.compute_k_style(img_tensor, 1)
                if new_class_code is None:
                    new_class_code = class_code
                else:
                    new_class_code += class_code
        return new_class_code / len(images)


    def transform_face_multi(self, input_img:np.ndarray) -> torch.Tensor:
        # This totally doesn't work.
        w = input_img.shape[0]
        if w < 256:
            return self.transform_face_1(input_img)
        else:
            split = w // 2
            tile = [[None, None],[None,None]]
            for i in range(2):
                for j in range(2):
                    sub = input_img[
                        i*split : i*split+split, 
                        j*split : j*split+split
                    ]
                    tile[i][j] = self.transform_face_1(sub)
            # rebuild
            for i in range(2):
                tile[i] = torch.cat(tile[i], dim=1)
            tiled = torch.cat(tile, dim=2)
            return tiled

                    
    def transform_face(self, input_img:np.ndarray) -> torch.Tensor:
        self.face_transform_cnt += 1
        return self.transform_face_1(input_img)

    @timebudget
    def transform_face_1(self, input_img:np.ndarray) -> torch.Tensor:
        image = Image.fromarray(input_img)
        content_img = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output_image = self.trainer.model.translate_simple(content_img, self.get_target_embedding())
            image = output_image.squeeze()
            image = (image + 1) / 2  # from (-1,1) to (0,1)
            return image

    def get_target_embedding(self) -> torch.Tensor:
        return self.target_embedding * self.scale_embedding


    def add_extra_faces(self, faces:np.ndarray) -> np.ndarray:
        """If "extra_detail" is configured, then it adds extra face box(es) so
        the CNN does a second (refiner) pass on the middle of the face.
        This adds amazing detail to tiger, but maybe not so good for others.
        """
        if not self.extra_detail:
            return faces
        assert isinstance(faces, np.ndarray)
        out = []
        for n in range(faces.shape[0]):
            f = faces[n,:]
            out.append(f)
            #TODO: generalize 640x480
            detail = list(self.resize_facebox(*f, (480,640), -0.2))  # smaller box
            out.append(list(detail))
            if self.extra_detail == 2:
                # add a second shifted down a titc.
                detail[1] += int(detail[3] * 0.5)
                out.append(detail)
        return out


    def resize_facebox(self, x:int, y:int, w:int, h:int, shape:Tuple, growth:float) -> Tuple:
        """Adjusts the facebox to be bigger (growth>0) or smaller (growth<0), and ensures it's a square
        that fits in the frame.
        """
        grow = int(w * growth)
        up_grow = int(grow * 0.7)
        nx = max(0, x-grow)
        ny = max(0, y-up_grow)
        nw = min(nx+w+2*grow, shape[1]) - nx
        nh = min(ny+h+2*grow, shape[0]) - ny
        #print(f"Old max is {x+w},{y+h}.  New is {nx+nw},{ny+nh}.  Grown by {grow}")
        nh = nw = min(nw,nh)
        return (nx, ny, nw, nh)


    def process_image(self, jpeg:bytes, save_file:str='output/face.jpg') -> np.ndarray:
        nparr = np.fromstring(jpeg, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.process_npimage(img_np, save_file)

    @timebudget
    def find_faces(self, img_np:np.ndarray) -> np.ndarray:
        bboxes = detect_faces(self.face_detect_model, img_np, minscale=2, ovr_threshhold=0.3, score_threshhold=0.5)
        out = []
        for bb in bboxes:
            x1, y1, x2, y2, _score = bb
            out.append([x1, y1, (x2-x1), (y2-y1)])
        return np.asarray(out)

    @timebudget
    def process_npimage(self, img_np:np.ndarray, save_file:str='output/face.jpg') -> np.ndarray:
        # or in torch...
        #pil_img = PILImage.open(io.BytesIO(jpeg))
        #img_tensor = ToTensor()(pil_img)
        faces = self.find_faces(img_np)
        if isinstance(faces,tuple):
            assert len(faces) == 0
            print("No faces found")
        else:
            faces = self.add_extra_faces(faces)
            num_faces = len(faces)
            if num_faces > self.max_faces:
                print(f"Too many faces {len(faces)}.  Pruning")
                faces = sorted(faces, key=lambda f: f[2], reverse=True)
                faces = np.asarray(faces[:self.max_faces])
                num_faces = len(faces)
            for face_num in range(num_faces):
                ox, oy, ow, oh = faces[face_num]
                if ow < self.min_face_size:
                    print(f"face at {ox},{oy} is {ow}x{oh} too small.  skipping")
                    continue
                x, y, w, h = self.resize_facebox(ox,oy,ow,oh, img_np.shape, self.grow_facebox)
                print(f"Found {w}x{h}px face at {x},{y}", end=" ")
                sub_img = img_np[y:y+h, x:x+w]
                xformed128 = self.transform_face(sub_img)
                self.blend_merge(img_np, xformed128, x, y, w, h)
        if save_file:
            cv2.imwrite(save_file, img_np)
            print(f"Saved to {save_file}")
        return img_np

    @timebudget
    def blend_merge(self, base:np.ndarray, face128:torch.Tensor, x:int, y:int, w:int, h:int):
        """Take the 128x128 transformed image, and resize it and blend it back into the 
        original in place."""
        xforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((h,w)),
            torchvision.transforms.ToTensor(),
        ])
        face = xforms(face128.cpu())
        face = mbcuda(face)
        face = face.permute(1,2,0) # CHW -> HWC
        face *= 255
        face = face[:, :, [2,1,0]]  # BGR to RGB
        face = self.mod_colors(face)
        alpha = self.prepare_alpha_mask_pt(h)
        old = mbcuda(torch.Tensor(base[y:y+h, x:x+w]))
        blended = old * (1-alpha) + face * alpha
        base[y:y+h, x:x+w] = blended.cpu().numpy()

    def mod_colors(self, face:torch.Tensor) -> torch.Tensor:
        return face * self.colorshift

    @functools.lru_cache(maxsize=1000)
    @timebudget
    def prepare_alpha_mask_pt(self, h:int, alpha_clamp:float=0.5) -> torch.Tensor:
        """alpha_clamp # Smaller numbers mean harsher boarders, but using more of the generated image
        """
        # Some heuristic math to come up with an alpha mask to apply to the image before pasting it back in
        line = mbcuda(torch.arange(-1, 1, 2/h, dtype=torch.float32).unsqueeze(0))
        assert len(line.shape) == 2  # see https://github.com/pytorch/pytorch/issues/28347
        line = line[:,0:h]
        assert line.shape == (1,h)
        alpha = line.T + line
        assert len(alpha.shape) == 2
        alpha = torch.abs(alpha) + torch.abs(torch.rot90(alpha))
        # Pretty much all of these constants can be tweaked to change how blending looks.
        alpha = torch.exp( - ((alpha/3)**2) * 5)
        alpha = (alpha - alpha.min()) ** 0.8
        alpha = torch.clamp(alpha, 0, alpha_clamp) / alpha_clamp * self.max_alpha
        alpha = alpha.unsqueeze(2).repeat(1,1,3)
        return alpha


class RoundRobinSpookifier(Spookifier):

    def __init__(self, target_image_base:str, target_classes:[str], noise_drift:float=0.1,
                 noise_speed:float=0.4, noise_mag:float=5, **kwargs):
        target_dirs = target_classes.split(',')
        target_dir = os.path.join(target_image_base, target_dirs[0])  # Any to init base class
        super().__init__(target_image_folder=target_dir, **kwargs)
        self._targets = []
        self._target_names = []
        for target in target_dirs:
            target_dir = os.path.join(target_image_base, target)
            embed = self.target_embedding_from_images(target_dir)
            print(f"Embedding range for {target} is {float(embed.min()):.2f} to {float(embed.max()):.2f}")
            self._targets.append(embed)
            self._target_names.append(target)
        self.noise_drift = noise_drift
        self.noise_speed = noise_speed
        self.noise_mag = noise_mag
        self.noise = self.noise_mag * torch.randn_like(self._targets[0])

    def get_target_embedding(self, current_time:float=None) -> torch.Tensor:
        if current_time is None:
            current_time = time.time()
        num = len(self._targets)
        turns = (current_time - 1.57e9) / self.cycle_delay  # subtract a base to reduce rounding errors
        alpha = turns - int(turns)  # for blending
        n = int(turns)
        # Do a simple randomish shuffle so target class order is pseudo-random
        first = ((n * 859) % 241) % num
        second = (((n+1) * 859) % 241) % num
        embed = self._targets[first]
        noise_mag = math.sin( turns * 6.28 * self.noise_speed )
        if alpha > 0:
            embed2 = self._targets[ second ]
            embed = embed2 * alpha + embed * (1-alpha)
            print(f"{100*alpha:.0f}% {self._target_names[second]} +{100*(1-alpha):.0f}% {self._target_names[first]} {noise_mag:+.2f}noise")
        else:
            print(f"Embedding is 100% {self._target_names[first]} with {noise_mag:.3f} noise")
        self.noise += self.noise_drift * torch.randn_like(self.noise)
        self.noise = torch.clamp(self.noise, -self.noise_mag, self.noise_mag)
        embed += self.noise * noise_mag
        embed *= self.scale_embedding
        return embed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', 
        type=str, 
        help='Image file to process')
    parser.add_argument('-t', '--target_image_folder',
        type=str,
        default='target-images/meerkat',
        help='Folder with examples of the target class')
    parser.add_argument('-b', '--box_expand',
        type=float,
        default=0.3,
        help='factor to increase size of facebox by')
    parser.add_argument('-t2', '--target2_image_folder',
        type=str,
        default=None,
        help='Optional folder with second target class to blend to')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    jpeg = open(args.input_file, "rb").read()
    spook = Spookifier(target_image_folder=args.target_image_folder, grow_facebox=args.box_expand)
    if args.target2_image_folder:
        target_embed = spook.target_embedding
        target2_embed = spook.target_embedding_from_images(args.target2_image_folder)
        for n, blend in enumerate(list(np.arange(0,1.01,0.1))):
            spook.target_embedding = target2_embed * blend + target_embed * (1-blend)
            spook.process_image(jpeg, f"output/blend-{n}.jpg")
    else:
        spook.process_image(jpeg, "output/spooky.jpg")

