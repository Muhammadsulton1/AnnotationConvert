from abc import ABC, abstractmethod
import os
import json
import tqdm
import numpy as np


class ConvertStrategy(ABC):
    @abstractmethod
    def convert(self, input_path, output_path):
        pass


class Coco2YoloStrategy(ConvertStrategy):
    def convert(self, input_path, output_path):
        pass

    def convert_without_segment(self, input_path, output_path):
        pass

    def convert_with_segment(self, input_path, output_path):
        pass


class Yolo2CocoStrategy(ConvertStrategy):
    def convert(self, input_path, output_path):
        pass

    def convert_without_segment(self, input_path, output_path):
        pass

    def convert_with_segment(self, input_path, output_path):
        pass


class Converter:
    def __init__(self, strategy: ConvertStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ConvertStrategy):
        self._strategy = strategy

    def convert(self, input_path, output_path):
        return self._strategy.convert(input_path, output_path)
