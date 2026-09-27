"""
Motion models for 64x64 pseudo-DVS frames (rps/dvs_emulator.py) and the shared class names.

RoshamboNet: our trainable network (train.py). DextraRoshamboNet: Dextra's pretrained network
(tools/import_dextra.py). Both keep the spatial progression of dextra-roshambo-python; ours adds
BatchNorm after every convolution.

RoshamboNet architecture:
  Input: (B, 1, 64, 64)
  Block 1: Conv(5x5, 16) -> BN -> ReLU -> AvgPool(2,2)   => (B, 16, 30, 30)
  Block 2: Conv(3x3, 32) -> BN -> ReLU -> AvgPool(2,2)   => (B, 32, 14, 14)
  Block 3: Conv(3x3, 64) -> BN -> ReLU -> AvgPool(2,2)   => (B, 64,  6,  6)
  Block 4: Conv(3x3,128) -> BN -> ReLU -> AvgPool(2,2)   => (B,128,  2,  2)
  Block 5: Conv(1x1,128) -> BN -> ReLU -> AvgPool(2,2)   => (B,128,  1,  1)
  FC:      Flatten -> Dropout(0.1) -> Linear(128, 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Class label mappings matching dataset folder names
CLASS_NAMES = ["0_rock", "1_paper", "2_scissors", "3_background"]
LABEL_TO_SYMBOL = {0: "rock", 1: "paper", 2: "scissors", 3: "background"}
SYMBOL_TO_LABEL = {v: k for k, v in LABEL_TO_SYMBOL.items()}

# The move that beats each human gesture (what the robot shows)
COUNTER_MOVES = {0: "paper", 1: "scissors", 2: "rock"}


class RoshamboNet(nn.Module):
    """
    Tiny 64x64 CNN (about 115k parameters) for pseudo-DVS frames; about 1-2 ms per frame on the CPU
    (measured by play.py). BatchNorm per block stabilizes training on sparse event frames.
    """
    def __init__(self, num_classes=4, pooling="avg", dropout=0.1):
        super(RoshamboNet, self).__init__()

        Pool = nn.AvgPool2d if pooling == "avg" else nn.MaxPool2d

        # Block 1: 64x64 -> 60x60 -> 30x30
        self.conv1 = nn.Conv2d(1,   16,  kernel_size=5, padding=0)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = Pool(2, 2)

        # Block 2: 30x30 -> 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16,  32,  kernel_size=3, padding=0)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = Pool(2, 2)

        # Block 3: 14x14 -> 12x12 -> 6x6
        self.conv3 = nn.Conv2d(32,  64,  kernel_size=3, padding=0)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = Pool(2, 2)

        # Block 4: 6x6 -> 4x4 -> 2x2
        self.conv4 = nn.Conv2d(64,  128, kernel_size=3, padding=0)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool4 = Pool(2, 2)

        # Block 5: 2x2 -> 2x2 -> 1x1
        self.conv5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.bn5   = nn.BatchNorm2d(128)
        self.pool5 = Pool(2, 2)

        # Classifier head
        self.drop = nn.Dropout(p=dropout)
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


class DextraRoshamboNet(nn.Module):
    """
    PyTorch port of Dextra's Keras RoshamboNet (SensorsINI/dextra-roshambo-python, deployed with
    pooling="avg"): same spatial progression as RoshamboNet but no BatchNorm and no dropout.
    Weights come from tools/import_dextra.py. Output classes are in DEXTRA_CLASS_ORDER.
    """
    def __init__(self, num_classes=4):
        super(DextraRoshamboNet, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=5), nn.Conv2d(16, 32, kernel_size=3), nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3), nn.Conv2d(128, 128, kernel_size=1),
        ])
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        for conv in self.convs:
            x = F.avg_pool2d(F.relu(conv(x)), 2)
        return self.fc(torch.flatten(x, 1))


# Dextra's output order (globals_and_utils.py SYMBOL_TO_PRED_DICT) and its mapping onto CLASS_NAMES
DEXTRA_CLASS_ORDER = ["paper", "scissors", "rock", "background"]
DEXTRA_TO_OURS = [DEXTRA_CLASS_ORDER.index(LABEL_TO_SYMBOL[i]) for i in range(4)]   # probs_ours = probs[DEXTRA_TO_OURS]


class MajorityVote:
    """
    Exact port of majority_vote temporal filter from dextra-roshambo-python consumer.py.
    Provides temporal stabilization over a 5-frame sliding window.
    """
    def __init__(self, window_length=5, num_classes=4):
        self.window_length = window_length
        self.num_classes   = num_classes
        self.ptr           = 0
        self.cirbuf        = [-1] * window_length
        self.cmdcnts       = [0]  * num_classes
        self.num_predictions = 0

    def new_prediction_and_vote(self, symbol: int):
        if 0 <= symbol < self.num_classes:
            self.num_predictions += 1
            idx = self.ptr
            if self.num_predictions > self.window_length:
                old = self.cirbuf[idx]
                if old >= 0:
                    self.cmdcnts[old] -= 1
            self.cirbuf[idx] = symbol
            self.cmdcnts[symbol] += 1
            self.ptr = (self.ptr + 1) % self.window_length
        return self.vote()

    def vote(self):
        majority_count = self.window_length // 2 + 1  # 3 for window=5
        max_idx = int(max(range(self.num_classes), key=lambda i: self.cmdcnts[i]))
        return max_idx if self.cmdcnts[max_idx] >= majority_count else None
