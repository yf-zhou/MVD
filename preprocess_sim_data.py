import torch 
from torchvision.io import decode_image, write_png
from torchvision.utils import save_image

import os

def process(image):
    image = torch.rot90(image, dims=(1, 2))
    image = image[:, 50:250, 70:275]
    return image

if __name__ == '__main__':
    src_dir = "data/11/sim"
    dst_dir = "data/11/sim_fixed"

    filenames = os.listdir(src_dir)

    print()

    for i, filename in enumerate(filenames):
        print(f"\rimage #{i:04}", end='')
        image = decode_image(os.path.join(src_dir, filename))

        image = process(image)

        write_png(image, os.path.join(dst_dir, filename))

    print()