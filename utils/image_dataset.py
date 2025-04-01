import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])

        width, height = img.size
        if width * height > 2**30:
            scale = (2**15 / max(width, height))
            new_size = (int(width*scale), int(height*scale))
            img = img.resize(new_size, Image.LANCZOS)

        return img

