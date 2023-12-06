import numpy as np
import os
import torch
from torch.utils.data import Dataset


COORDINATES_FILE_PATH = "/meta/neurons/cell_motor_coordinates.npy"
RESPONSES_PATH = "/data/responses/"
VIDEOS_PATH = "/data/videos/"


def get_nonzero_count(input: torch.Tensor) -> int:
    (x, y, z) = input.shape
    indices = [(xx, yy, zz) for xx in range(x) for yy in range(y) for zz in range(z)]
    result = 0
    for xx, yy, zz in indices:
        if input[xx][yy][zz] > 0:
            result += 1
    return result


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


class NeurosismDataset(Dataset):
    def __init__(self, overlayCount, parentDirectory, poolFactor, densify=False, frameSkipCount=50, seed=None):
        self.overlayCount = overlayCount
        self.coordinates: np.ndarray = np.load(parentDirectory + COORDINATES_FILE_PATH)
        self.responsePath = parentDirectory + RESPONSES_PATH
        self.videoPath = parentDirectory + VIDEOS_PATH
        self.frameSkipCount = frameSkipCount
        self.poolFactor = poolFactor if type(poolFactor) == tuple else (poolFactor, poolFactor, poolFactor)
        self.rng = np.random.default_rng(seed=seed)

        self.neuronCount = self.coordinates.shape[0]
        if densify:
            self.coordinates = self._get_condensed_coordinates()
        else:
            self._get_condensed_coordinates()
        self._calculate_data_dimensions()
        self.validFiles = get_valid_files(self.responsePath, self.videoPath)

    def __len__(self):
        return len(self.validFiles)

    def __getitem__(self, idx):
        fileName = self.validFiles[idx]
        responses = np.load(self.responsePath + fileName)
        video: np.ndarray = np.load(self.videoPath + fileName)
        nthFrame = self.rng.integers(self.overlayCount + self.frameSkipCount, video.shape[2])

        image = self._get_image(video, nthFrame)
        image = torch.tensor(image).float()

        data = np.zeros((1, self.zRange + 1, self.yRange + 1, self.xRange + 1))
        # data = np.zeros((1, self.xRange + 1, self.yRange + 1, self.zRange + 1))
        for n in range(self.neuronCount):
            responsesOverTime = responses[n]
            accumulator = 0
            for t in range(nthFrame - self.overlayCount, nthFrame):
                accumulator += responsesOverTime[t]
                accumulator /= 2
            (x, y, z) = self.coordinates[n]
            x = int((x - self.xMin) / self.poolFactor[2])
            y = int((y - self.yMin) / self.poolFactor[1])
            z = int((z - self.zMin) / self.poolFactor[0])
            data[0][z][y][x] = accumulator
            # x = int((x - self.xMin) / self.poolFactor[0])
            # y = int((y - self.yMin) / self.poolFactor[1])
            # z = int((z - self.zMin) / self.poolFactor[2])
            # data[0][x][y][z] = accumulator
        data = torch.tensor(data).float()

        return data, image

    def _get_condensed_coordinates(self):
        xArray = yArray = zArray = np.ndarray(0, dtype=np.uint16)
        for n in range(self.neuronCount):
            x, y, z = self.coordinates[n]
            if x not in xArray:
                xArray = np.append(xArray, x)
            if y not in yArray:
                yArray = np.append(yArray, y)
            if z not in zArray:
                zArray = np.append(zArray, z)
        self.xCount = len(xArray)
        self.yCount = len(yArray)
        self.zCount = len(zArray)

        xArray = np.sort(xArray)
        yArray = np.sort(yArray)
        zArray = np.sort(zArray)
        xMap = yMap = zMap = {}
        for i in range(self.xCount):
            xMap[xArray[i]] = i
        for i in range(self.yCount):
            yMap[yArray[i]] = i
        for i in range(self.zCount):
            zMap[zArray[i]] = i

        coordinates = np.zeros((self.neuronCount, 3), dtype=np.uint16)
        for n in range(self.neuronCount):
            x, y, z = self.coordinates[n]
            coordinates[n] = xMap[x], yMap[y], zMap[z]
        return coordinates

    def _calculate_data_dimensions(self):
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
        self.xRange = int(1 + (xMax - self.xMin) / self.poolFactor[2])
        self.yRange = int(1 + (yMax - self.yMin) / self.poolFactor[1])
        self.zRange = int(1 + (zMax - self.zMin) / self.poolFactor[0])
        # self.xRange = int(1 + (xMax - self.xMin) / self.poolFactor[0])
        # self.yRange = int(1 + (yMax - self.yMin) / self.poolFactor[1])
        # self.zRange = int(1 + (zMax - self.zMin) / self.poolFactor[2])

    def _get_image(self, video: np.ndarray, nthFrame) -> np.ndarray:
        (h, w, _) = video.shape
        result = np.ndarray((1, 1, h, w))
        indices = [(hh, ww) for hh in range(h) for ww in range(w)]
        for hh, ww in indices:
            result[0][0][hh][ww] = video[hh][ww][nthFrame]
        return result
