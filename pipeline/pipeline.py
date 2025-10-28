from typing import List
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

import logging

import numpy as np

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class StageBase(ABC, Generic[InputType, OutputType]):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self, input: InputType) -> OutputType:
        raise NotImplementedError

@dataclass
class VideoConfiguration:
    name : str
    path: Path
    scale_factor: float | None

@dataclass
class VideoData:
    frames: np.ndarray
    fps: float


class BoxingDynamicsPipeline:
    def __init__(self, stages: List[StageBase]):
        self.stages = stages
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, input: VideoConfiguration):
        data = input
        for stage in self.stages:
            self.logger.info(f"Pipeline starting executing: {stage.__class__.__name__}")
            data = stage.execute(data)
            self.logger.info(f"Pipeline finished executing: {stage.__class__.__name__}")
        return data
