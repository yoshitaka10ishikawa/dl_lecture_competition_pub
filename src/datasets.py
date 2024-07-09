import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

### NOTE: 追加ライブラリ
import torch.nn.functional as F
from scipy.signal import resample, butter, filtfilt

### NOTE: 脳波（EEG）データの前処理
### リサンプリング: 元のサンプリング周波数（1000 Hzと仮定）から新しい周波数（デフォルトで100 Hz）にデータをリサンプリングする。
### バンドパスフィルタリング: バンドパスフィルタを適用して、1 Hzから40 Hzの周波数帯の信号を抽出する。バターワースフィルタを使用する。
### 正規化: zスコア正規化を行い、データを平均0、標準偏差1の分布に変換する。
def preprocess_eeg(data, fs_new=100):
    # Original sampling frequency (assuming 1000 Hz)
    fs = 1000
    
    # Resample to new frequency
    data_resampled = resample(data, int(data.shape[-1] * fs_new / fs), axis=-1)
    
    # Apply a bandpass filter (1-40 Hz)
    nyquist = 0.5 * fs_new
    low = 1 / nyquist
    high = 40 / nyquist
    b, a = butter(1, [low, high], btype='band')
    data_filtered = filtfilt(b, a, data_resampled, axis=-1)
    
    # Normalize (z-score)
    data_normalized = (data_filtered - np.mean(data_filtered, axis=-1, keepdims=True)) / np.std(data_filtered, axis=-1, keepdims=True)
    
    return data_normalized


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]