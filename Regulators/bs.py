from audioop import mul
import numpy as np

class BSStandart : 

    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def adjust(self, lr):
        lr *= self.multiplier
        return lr