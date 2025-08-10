import av
import os
import pims
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class VideoReader(Dataset):
    def __init__(self, path, max_frames=None, transform=None):
        self.video = pims.PyAVVideoReader(path)
        self.rate = self.video.frame_rate
        self.transform = transform
        self.max_frames = max_frames
        
    @property
    def frame_rate(self):
        return self.rate
    
    @property
    def origin_shape(self):
        return self.video[0].shape[:2]

    def __len__(self):
        if self.max_frames is not None and self.max_frames > 0:
            return min(len(self.video), self.max_frames)
        else:
            return len(self.video)
        
    def __getitem__(self, idx):
        frame = self.video[idx]
        frame = Image.fromarray(np.asarray(frame))
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        self.container = av.open(path, mode='w')
        # self.container.add_stream('h264', rate=30)
        self.stream = self.container.add_stream('h264', rate=f'{frame_rate:.4f}')
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate
    
    def write(self, frames):

        # frames: [T, C, H, W]
        self.stream.width = frames.size(3)
        self.stream.height = frames.size(2)
        if frames.size(1) == 1:
            frames = frames.repeat(1, 3, 1, 1) # convert grayscale to RGB
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()

        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            self.container.mux(self.stream.encode(frame))

    def write_numpy(self, frames):
        
        # frames: [T, H, W, C]
        self.stream.height = frames.shape[1]
        self.stream.width = frames.shape[2]

        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            self.container.mux(self.stream.encode(frame))

    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()


class ImageSequenceReader(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = sorted(os.listdir(path))
        self.transform = transform

    @property
    def origin_shape(self):
        return np.array(Image.open(os.path.join(self.path, self.files[0]))).shape[:2]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
            img.load()
        
        origin_shape = torch.from_numpy(np.asarray(np.array(img).shape[:2]))

        if self.transform is not None:
            img, filename = self.transform(img), self.files[idx]
        else:
            filename = self.files[idx]

        return {"image": img, "filename": filename, "origin_shape": origin_shape}


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)
    
    def write(self, frames, filenames=None):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            if filenames is None:
                filename = str(self.counter).zfill(4) + '.' + self.extension
            else:
                filename = filenames[t].split('.')[0] + '.' + self.extension

            to_pil_image(frames[t]).save(os.path.join(
                self.path, filename))
            self.counter += 1
            
    def close(self):
        pass
        