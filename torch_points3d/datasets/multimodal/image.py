import os.path as osp
import numpy as np
from PIL import Image



class ImageData(object):

    def __init__(self, path, pos, opk, size=(2048, 1024)):
        self.pos = pos
        self.opk = opk
        self.size = size
        self.path = path
        self.name = osp.splitext(osp.basename(path))[0]
        self.image = self.read_image(self.path, self.size)


    @staticmethod
    def read_image(path, size):
        return np.array(Image.open(path).convert('RGB').resize(size, Image.LANCZOS))
