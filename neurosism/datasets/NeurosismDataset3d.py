import numpy as np

import os

import torch
from torch.utils.data import Dataset

# from torch import Tensor


FRAME_SKIP_COUNT = 50
# POOL_FACTOR = 4
# POOL_FACTOR = 8
# POOL_FACTOR = 16  # 5528
# POOL_FACTOR = 32  # 2496
# POOL_FACTOR = 64 # 394

COORDINATES_FILE_PATH = "/meta/neurons/cell_motor_coordinates.npy"
RESPONSES_PATH = "/data/responses/"
VIDEOS_PATH = "/data/videos/"


def get_valid_files(filePath1: str, filePath2: str) -> list:
    fileNames1 = os.listdir(filePath1)
    fileNames2 = os.listdir(filePath2)
    result = []
    for f in fileNames1:
        if f in fileNames2:
            result.append(f)
    return result


def reshape_video_hwd_to_dhw(video: np.ndarray) -> np.ndarray:
    (height, width, duration) = video.shape
    indices = [(d, h, w) for d in range(duration) for h in range(height) for w in range(width)]
    result = np.zeros((duration, height, width))
    for d, h, w in indices:
        result[d][h][w] = video[h][w][d]
    return result


class NeurosismDataset3d(Dataset):
    def __init__(self, device, frameCount, parentDirectory, densify=False, pool=1, seed=None):
        self.device = device
        self.frameCount = frameCount
        self.coordinates: np.ndarray = np.load(parentDirectory + COORDINATES_FILE_PATH)
        self.responsePath = parentDirectory + RESPONSES_PATH
        self.videoPath = parentDirectory + VIDEOS_PATH
        self.pool = pool
        self.rng = np.random.default_rng(seed=seed)

        self.neuronCount = self.coordinates.shape[0]
        if densify:
            self.coordinates = self._condense_coordinates()
        self._generate_data_dimensions()
        self.validFiles = get_valid_files(self.responsePath, self.videoPath)

    def __len__(self):
        return len(self.validFiles)

    def __getitem__(self, idx):
        fileName = self.validFiles[idx]

        responses = np.load(self.responsePath + fileName)
        video: np.ndarray = np.load(self.videoPath + fileName)
        nthFrame = self.rng.integers(self.frameCount + FRAME_SKIP_COUNT, video.shape[2])
        image = self._get_image(video, nthFrame)

        data = np.zeros((1, self.xRange, self.yRange, self.zRange))
        for n in range(self.neuronCount):
            (x, y, z) = self.coordinates[n]
            x = (x - self.xMin) // self.pool
            y = (y - self.yMin) // self.pool
            z = (z - self.zMin) // self.pool
            responsesOverTime = responses[n]
            accumulator = 0
            for t in range(nthFrame - self.frameCount, nthFrame):
                accumulator += responsesOverTime[t]
                accumulator /= 2
            data[0][x][y][z] = accumulator
        return data, image

    def _condense_coordinates(self):
        xArray = yArray = zArray = np.ndarray(0, dtype=np.uint16)
        for n in range(self.neuronCount):
            x, y, z = self.coordinates[n]
            if x not in xArray:
                xArray = np.append(xArray, x)
            if y not in yArray:
                yArray = np.append(yArray, y)
            if z not in zArray:
                zArray = np.append(zArray, z)
        xCount = len(xArray)
        yCount = len(yArray)
        zCount = len(zArray)
        coordinates = np.zeros((self.neuronCount, 3), dtype=np.uint16)

        xArray = np.sort(xArray)
        yArray = np.sort(yArray)
        zArray = np.sort(zArray)
        xMap = yMap = zMap = {}
        for i in range(xCount):
            xMap[xArray[i]] = i
        for i in range(yCount):
            yMap[yArray[i]] = i
        for i in range(zCount):
            zMap[zArray[i]] = i
        for n in range(self.neuronCount):
            x, y, z = self.coordinates[n]
            coordinates[n] = xMap[x], yMap[y], zMap[z]
        return coordinates

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
        self.xRange = 1 + (xMax - self.xMin) // self.pool
        self.yRange = 1 + (yMax - self.yMin) // self.pool
        self.zRange = 1 + (zMax - self.zMin) // self.pool

    def _get_image(self, video: np.ndarray, nthFrame):
        (h, w, _) = video.shape
        result = np.ndarray((1, h, w))
        indices = [(hh, ww) for hh in range(h) for ww in range(w)]
        for hh, ww in indices:
            result[0][hh][ww] = video[hh][ww][nthFrame]
        return result

    def _reshape_data(self, data: np.ndarray) -> np.ndarray:
        (time, x, y, z) = data.shape
        result = np.zeros((z, x * y, time))
        indices = [(t, xx, yy, zz) for t in range(time) for xx in range(x) for yy in range(y) for zz in range(z)]
        for t, xx, yy, zz in indices:
            result[zz][y * xx + yy][t] = data[t][xx][yy][zz]
        return result

    def _reshape_video(self, video: np.ndarray) -> np.ndarray:
        (height, width, time) = video.shape
        assert self.frameCount <= time
        result = np.zeros((1, height * width, self.frameCount))
        indices = [(t, h, w) for t in range(self.frameCount) for h in range(height) for w in range(width)]
        for t, h, w in indices:
            result[0][width * h + w][t] = video[h][w][t + FRAME_SKIP_COUNT]
        return result
