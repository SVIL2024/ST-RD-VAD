import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data


def np_load_frame(filename, resize_height, resize_width, grayscale=False):
    """
    Load image and normalize for pretrained I3D.
    - cv2.imread: BGR -> RGB
    - [0, 255] -> [-1, 1]
    """
    if grayscale:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (resize_width, resize_height)).astype(np.float32)
        image = (image / 127.5) - 1.0
        return image
    else:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (resize_width, resize_height)).astype(np.float32)
        image = (image / 127.5) - 1.0
        return image


class Reconstruction3DDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, num_frames=16,
                 img_extension='.jpg', dataset='ped2', jump=None, train=True, train_stride=1,
                 video_names=None, pseudo_seed=0, deterministic_pseudo=False):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._num_frames = num_frames
        self.jump = [2] if jump is None else jump
        self.extension = img_extension
        self.dataset = dataset
        self.train_stride = train_stride if train else 1
        self.train = train
        self.allowed_video_names = None if video_names is None else set(video_names)
        self.pseudo_seed = int(pseudo_seed)
        self.deterministic_pseudo = bool(deterministic_pseudo)
        self.gray = False
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*/'))
        for video in sorted(videos):
            video_name = os.path.basename(os.path.normpath(video))
            if self.allowed_video_names is not None and video_name not in self.allowed_video_names:
                continue
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.extension))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        for video_name in self.videos:
            for i in range(0, len(self.videos[video_name]['frame']) - self._num_frames + 1, self.train_stride):
                frames.append(self.videos[video_name]['frame'][i])
        return frames

    def __getitem__(self, index):
        video_name = os.path.basename(os.path.dirname(self.samples[index]))
        frame_name = int(os.path.splitext(os.path.basename(self.samples[index]))[0]) - 1

        batch = []
        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i],
                                  self._resize_height, self._resize_width, grayscale=self.gray)
            if self.transform is not None:
                batch.append(self.transform(image))

        clip = np.stack(batch, axis=1)

        if self.train:
            rng = (
                np.random.RandomState(self.pseudo_seed + int(index))
                if self.deterministic_pseudo
                else np.random
            )
            shuffle_idx = rng.permutation(self._num_frames)
            pseudo_clip = clip[:, shuffle_idx, :, :]
            return clip, pseudo_clip
        else:
            return clip

    def __len__(self):
        return len(self.samples)
