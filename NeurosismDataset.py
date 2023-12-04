import numpy as np

import os

import torch
from torch.utils.data import Dataset


DEFAULT_FRAME_SKIP_COUNT = 50
POOL_FACTOR = 32


def reshape_video_hwd_to_dhw(video: np.ndarray) -> np.ndarray:
    (height, width, duration) = video.shape
    result = np.empty((duration, height, width))

    indices = [(d, h, w) for d in range(duration) for h in range(height) for w in range(width)]
    for d, h, w in indices:
        result[d][h][w] = video[h][w][d]

    return result


class NeurosismDataset(Dataset):
    def __init__(self, frameCount, parentDirectory, transform=None, transformTarget=None):
        self.frameCount = frameCount
        self.coordinates = np.load(parentDirectory + "/meta/neurons/cell_motor_coordinates.npy")
        self.responsePath = parentDirectory + "/data/responses/"
        self.videoPath = parentDirectory + "/data/videos/"
        self._generate_data_dimensions()

    def __len__(self):
        return len(os.listdir(self.videoPath))

    def __getitem__(self, idx):
        data = torch.zeros(self.frameCount - DEFAULT_FRAME_SKIP_COUNT, self.xRange, self.yRange, self.zRange)
        fileName = str(idx) + ".npy"
        responses = torch.tensor(np.load(self.responsePath + fileName))
        for n in range(self.neuronCount):
            (x, y, z) = self.coordinates[n]
            x = (x - self.xMin) // POOL_FACTOR
            y = (y - self.yMin) // POOL_FACTOR
            z = (z - self.zMin) // POOL_FACTOR
            responsesOverTime = responses[n]
            for t in range(self.frameCount - DEFAULT_FRAME_SKIP_COUNT):
                data[t][x][y][z] += responsesOverTime[t + DEFAULT_FRAME_SKIP_COUNT]
                # currentData = data[t][x][y][z]
                # incomingResponse = responsesOverTime[t + DEFAULT_FRAME_SKIP_COUNT]
                # # if n == 0:
                # #     print(incomingResponse)
                # if incomingResponse > currentData:
                #     # if n == 0:
                #     #     print("test")
                #     data[t][x][y][z] = incomingResponse
        # for t in range(self.frameCount - DEFAULT_FRAME_SKIP_COUNT):
        #     currentSlice = data[t]
        #     # incomingSlice = responses[t + DEFAULT_FRAME_SKIP_COUNT]
        #     for n in range(self.neuronCount):
        #         (x, y, z) = self.coordinates[n]
        #         x -= self.xMin
        #         y -= self.yMin
        #         z -= self.zMin
        #         xPool = x // POOL_FACTOR
        #         yPool = y // POOL_FACTOR
        #         zPool = z // POOL_FACTOR
        #         currentData = currentSlice[xPool][yPool][zPool]
        #         incomingData = incomingSlice[x][y][z]
        #         if incomingData > currentData:
        #             data[t][xPool][yPool][zPool] = incomingData
        video = reshape_video_hwd_to_dhw(np.load(self.videoPath + fileName))
        # data = self.dataTemplate.copy()
        # for n in range(self.neuronCount):
        #     (x, y, z) = self.coordinates[n]
        #     responsesOverTime = responses[n]
        #     for t in range(len(responsesOverTime) - DEFAULT_FRAME_SKIP_COUNT):
        #         data[t][x - self.xMin][y - self.yMin][z - self.zMin] = responsesOverTime[t + DEFAULT_FRAME_SKIP_COUNT]
        return data, torch.tensor(video)

    def _generate_data_template(self, frameCount: int, frameSkipCount=DEFAULT_FRAME_SKIP_COUNT):
        self.xMin = xMax = 0
        self.yMin = yMax = 0
        self.zMin = zMax = 0
        self.neuronCount = self.coordinates.shape[0]
        for n in range(self.neuronCount):
            (x, y, z) = self.coordinates[n]
            xMax = x if x > xMax else xMax
            yMax = y if y > yMax else yMax
            zMax = z if z > zMax else zMax
            self.xMin = x if x < self.xMin else self.xMin
            self.yMin = y if y < self.yMin else self.yMin
            self.zMin = z if z < self.zMin else self.zMin
        xRange = xMax - self.xMin
        yRange = yMax - self.yMin
        zRange = zMax - self.zMin
        self.dataTemplate = torch.zeros(frameCount - frameSkipCount, xRange, yRange, zRange)

    def _generate_data_dimensions(self):
        self.xMin = xMax = 0
        self.yMin = yMax = 0
        self.zMin = zMax = 0
        self.neuronCount = self.coordinates.shape[0]
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
