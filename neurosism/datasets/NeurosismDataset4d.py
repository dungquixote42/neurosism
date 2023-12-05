import numpy as np

import os

import torch
from torch.utils.data import Dataset

# from torch import Tensor


FRAME_SKIP_COUNT = 50
# POOL_FACTOR = 2 # 6179
# POOL_FACTOR = 4 # 6179
# POOL_FACTOR = 8 # 6170
POOL_FACTOR = 16  # 5586
# POOL_FACTOR = 32  # 2590
# POOL_FACTOR = 64 # 398


def get_valid_files(filePath1: str, filePath2: str) -> list:
    fileNames1 = os.listdir(filePath1)
    fileNames2 = os.listdir(filePath2)
    result = []
    for f in fileNames1:
        if f in fileNames2:
            result.append(f)
    return result


def reshape_video_hwd_to_dhw(video: np.ndarray) -> np.ndarray:
    (height, width, time) = video.shape
    indices = [(t, h, w) for t in range(time) for h in range(height) for w in range(width)]
    result = np.zeros((time, height, width))
    for t, h, w in indices:
        result[t][h][w] = video[h][w][t]
    return result


class NeurosismDataset4d(Dataset):
    def __init__(self, device, frameCount, parentDirectory, transform=None, transform_target=None):
        self.device = device
        self.frameCountActual = frameCount - FRAME_SKIP_COUNT
        self.coordinates: np.ndarray = np.load(parentDirectory + "/meta/neurons/cell_motor_coordinates.npy")
        self.responsePath = parentDirectory + "/data/responses/"
        self.videoPath = parentDirectory + "/data/videos/"

        self.neuronCount = self.coordinates.shape[0]
        self._generate_data_dimensions()
        self.validFiles = get_valid_files(self.responsePath, self.videoPath)

    def __len__(self):
        return len(self.validFiles)

    def __getitem__(self, idx):
        fileName = self.validFiles[idx]

        responses = np.load(self.responsePath + fileName)
        video = np.load(self.videoPath + fileName)

        data = np.zeros((self.frameCountActual, self.xRange, self.yRange, self.zRange))
        for n in range(self.neuronCount):
            (x, y, z) = self.coordinates[n]
            x = (x - self.xMin) // POOL_FACTOR
            y = (y - self.yMin) // POOL_FACTOR
            z = (z - self.zMin) // POOL_FACTOR
            responsesOverTime = responses[n]
            for t in range(self.frameCountActual):
                current = data[t][x][y][z]
                incoming = responsesOverTime[t + FRAME_SKIP_COUNT]
                data[t][x][y][z] = incoming if incoming > current else current
        # data = self._reshape_data(data)
        # video = self._reshape_video(video)
        return data, video

    def _generate_data_dimensions(self):
        self.xMin = xMax = 0
        self.yMin = yMax = 0
        self.zMin = zMax = 0
        for n in range(self.neuronCount):
            (x, y, z) = self.coordinates[n]
            xMax = x if x > xMax else xMax
            yMax = y if y > yMax else yMax
            zMax = z if z > zMax else zMax
            self.xMin = x if x < self.xMin else self.xMin
            self.yMin = y if y < self.yMin else self.yMin
            self.zMin = z if z < self.zMin else self.zMin
        self.xRange = 1 + (xMax - self.xMin) // POOL_FACTOR
        self.yRange = 1 + (yMax - self.yMin) // POOL_FACTOR
        self.zRange = 1 + (zMax - self.zMin) // POOL_FACTOR

    def _reshape_data(self, data: np.ndarray) -> np.ndarray:
        (time, x, y, z) = data.shape
        result = np.zeros((z, x * y, time))
        indices = [(t, xx, yy, zz) for t in range(time) for xx in range(x) for yy in range(y) for zz in range(z)]
        for t, xx, yy, zz in indices:
            result[zz][y * xx + yy][t] = data[t][xx][yy][zz]
        return result

    def _reshape_video(self, video: np.ndarray) -> np.ndarray:
        (height, width, time) = video.shape
        assert self.frameCountActual <= time
        result = np.zeros((1, height * width, self.frameCountActual))
        indices = [(t, h, w) for t in range(self.frameCountActual) for h in range(height) for w in range(width)]
        for t, h, w in indices:
            result[0][width * h + w][t] = video[h][w][t + FRAME_SKIP_COUNT]
        return result
