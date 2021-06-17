"""
    Read feature file and mu-law file
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


class FeaturePCMLoader(Dataset):
    def __init__(self, feature_name, pcm_name, frame_size, nb_used_features, bfcc_band, nb_features, pitch_idx, n_samples_per_step, checkpoint_path):
        self.features = feature_name
        self.pcm = pcm_name
        self.frame_size = frame_size
        self.nb_features = nb_features  # NB_FEATURES (2*NB_BANDS+3+LPC_ORDER)
        self.nb_used_features = nb_used_features
        self.bfcc_band = bfcc_band
        self.pitch_idx   = pitch_idx
        self.n_samples_per_step = n_samples_per_step
        self.checkpoint_path = checkpoint_path
        self.in_data, self.features, self.periods, self.out_exc = self.process_feature_pcm(self.features, self.pcm, self.n_samples_per_step, self.checkpoint_path)

    def process_feature_pcm(self, feature_file, pcm_file, n_samples_per_step, checkpoint_path):

        frame_size = self.frame_size
        nb_features = self.nb_features  # NB_FEATURES (2*NB_BANDS+3+LPC_ORDER)
        bfcc_band = self.bfcc_band
        nb_used_features = self.nb_used_features
        feature_chunk_size = 15
        pcm_chunk_size = frame_size * feature_chunk_size

        data = np.fromfile(pcm_file, dtype='uint8')
        nb_frames = len(data) // (4 * pcm_chunk_size)

        features = np.fromfile(feature_file, dtype='float32')

        # limit to discrete number of frames
        data = data[:nb_frames * 4 * pcm_chunk_size]
        features = features[:nb_frames * feature_chunk_size * nb_features]
        features = np.reshape(features, (nb_frames * feature_chunk_size, nb_features))
        
        sig = data[0::4]
        pred = data[1::4]
        in_exc = data[2::4]
        out_exc = data[3::4]
        if self.n_samples_per_step > 1:
            pad_in_data = np.ones([self.n_samples_per_step - 1, 3], dtype=data.dtype)
            pad_in_data[..., 0] = pad_in_data[..., 0] * sig[0]
            pad_in_data[..., 1] = pad_in_data[..., 1] * pred[0]
            pad_in_data[..., 2] = pad_in_data[..., 2] * in_exc[0]
            sig = np.concatenate([pad_in_data[:, 0], sig], axis=0)[:len(sig)]
            pred = np.concatenate([pad_in_data[:, 1], pred], axis=0)[:len(pred)]
            in_exc = np.concatenate([pad_in_data[:, 2], in_exc], axis=0)[:len(in_exc)]

        sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
        pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
        in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
        out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))

        features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
        features = features[..., :nb_used_features]  # [nb_frames, 160*15, 38]
        features[..., bfcc_band:2*bfcc_band] = 0

        fpad1 = np.concatenate([features[0:1, 0:2, :], features[:-1, -2:, :]], axis=0)
        fpad2 = np.concatenate([features[1:, :2, :], features[0:1, -2:, :]], axis=0)
        features = np.concatenate([fpad1, features, fpad2], axis=1)

        periods = (.1 + 50 * features[..., self.pitch_idx:self.pitch_idx+1] + 100).astype('int16')

        del sig
        del pred
        del in_exc
        del data
        
        return in_data, features, periods, out_exc

    def __getitem__(self, index):
        return [torch.tensor(self.in_data[index], dtype=torch.long),
                torch.tensor(self.features[index], dtype=torch.float32),
                torch.tensor(self.periods[index], dtype=torch.long),
                torch.tensor(self.out_exc[index], dtype=torch.long)]

    def __len__(self):
        return len(self.in_data)
