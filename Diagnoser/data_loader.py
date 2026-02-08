import logging
import os
import traceback
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FusionDataset(Dataset):
    """Fusion dataset implementation - focused on respiratory sound waveform data"""

    def __init__(self, root_dir, config, target_device=None, metadata_path="audio_descriptions.jsonl"):
        """Initializes the dataset
        Args:
            root_dir: The root directory of the dataset.
            config: The configuration dictionary.
            target_device: The target device this dataset instance should use (e.g., 'cuda:local_rank').
            metadata_path: The metadata filename (e.g., 'audio_descriptions.jsonl').
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.config = config
        self.metadata_path = metadata_path

        if target_device is not None:
            self.processor_device = torch.device(target_device)
        else:
            local_rank_str = os.environ.get("LOCAL_RANK")
            if local_rank_str is not None:
                try:
                    self.processor_device = torch.device(f'cuda:{int(local_rank_str)}')
                except ValueError:
                    logger.warning(
                        f"FusionDataset: LOCAL_RANK ('{local_rank_str}') is invalid. Falling back to CPU for internal processor.")
                    self.processor_device = torch.device('cpu')
            else:
                logger.warning(
                    "FusionDataset: target_device not provided and LOCAL_RANK is not set. Falling back to CPU for internal processor.")
                self.processor_device = torch.device('cpu')
        logger.info(f"FusionDataset instance will configure device for internal processor: {self.processor_device}")

        if not self.root_dir.exists():
            raise ValueError(f"Dataset path does not exist: {self.root_dir}")

        logger.info(f"Dataset root directory: {self.root_dir}")

        self.eps = 1e-8
        self.max_audio_value = 1.0
        self.min_audio_value = -1.0

        # Load metadata once at initialization to get all valid audio filenames
        self.valid_filenames = self._load_valid_filenames_from_metadata()

        self.default_features = {
            'waveform': torch.zeros(self.config['audio']['max_length'], device=torch.device('cpu')),
            'file_path': "",
            'mel_lengths': torch.tensor(0, dtype=torch.long, device=torch.device('cpu'))
        }
        self._init_dataset()

    def _load_valid_filenames_from_metadata(self):
        """
        Loads all valid audio filenames from audio_descriptions.jsonl.
        Returns a set containing all filenames for fast lookup.
        """
        valid_filenames = set()
        jsonl_path = self.metadata_path

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        filename = record.get("audio_filename")
                        if filename:
                            valid_filenames.add(filename)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse JSON line, skipping: {line.strip()}")

            if not valid_filenames:
                raise ValueError("No valid 'audio_filename' found in the metadata file.")

            logger.info(f"Loaded {len(valid_filenames)} valid audio filenames from {jsonl_path}.")
            return valid_filenames

        except FileNotFoundError:
            logger.error(f"Metadata file not found: {jsonl_path}. Cannot build dataset.")
            raise
        except Exception as e:
            logger.error(f"Failed to load metadata file: {jsonl_path}, error: {e}")
            raise

    def _init_dataset(self):
        # Main sample rate, used for the final output waveform
        self.sample_rate = self.config['audio']['sample_rate']
        # Target sample rate, used only for calculating vocos mel length
        self.target_vocos_sr = self.config['vocos']['sample_rate']

        self.mel_hop_length = self.config['vocos']['mel']['hop_length']
        self.mel_n_fft = self.config['vocos']['mel']['n_fft']

        self.file_paths = self._get_file_paths()
        if not self.file_paths:
            raise ValueError(f"No audio files matching the metadata were found in the directory {self.root_dir}.")

    def _get_file_paths(self):
        """
        Gets all .wav audio file paths, but only those that exist in the metadata.
        """
        file_paths = []
        logger.info(f"Scanning {self.root_dir} and matching against {len(self.valid_filenames)} valid filenames...")

        # Scan for all .wav files in the directory
        all_wav_files = list(self.root_dir.glob("*.wav"))

        # Keep only those files whose names exist in the self.valid_filenames set
        for audio_file in all_wav_files:
            if audio_file.name in self.valid_filenames:
                file_paths.append(audio_file)

        if not file_paths:
            logger.warning(f"No .wav audio files matching the metadata list were found in {self.root_dir}")

        file_paths = sorted(file_paths)
        logger.info(
            f"Matching complete. Found {len(file_paths)} valid audio files in total. First one: {file_paths[0] if file_paths else 'N/A'}")
        return file_paths

    def _safe_load_audio(self, audio_path):
        try:
            audio_path_obj = Path(audio_path)
            if not audio_path_obj.is_file():
                raise FileNotFoundError(f"Audio file does not exist: {audio_path_obj}")

            waveform, sr = torchaudio.load(str(audio_path_obj))

            if waveform.numel() == 0:
                raise ValueError(f"Loaded audio is empty: shape={waveform.shape}, path={audio_path_obj}")

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            max_val = torch.abs(waveform).max()
            safe_divisor = max_val if max_val > self.eps else torch.tensor(1.0, device=waveform.device,
                                                                           dtype=waveform.dtype)
            waveform = waveform / safe_divisor
            waveform = torch.clamp(waveform, self.min_audio_value, self.max_audio_value)

            return waveform, sr
        except Exception as e:
            # logger.error(f"Failed to load audio file {audio_path}: {str(e)}\n{traceback.format_exc()}")
            return self.default_features['waveform'].clone().unsqueeze(0), self.sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            file_path = self.file_paths[idx]
            waveform_orig_cpu_batched, sr_orig = self._safe_load_audio(file_path)

            waveform_for_mel_calc_cpu_batched = waveform_orig_cpu_batched
            if sr_orig != self.target_vocos_sr:
                try:
                    waveform_for_mel_calc_cpu_batched = torchaudio.functional.resample(
                        waveform_orig_cpu_batched, sr_orig, self.target_vocos_sr
                    )
                except Exception as e:
                    logger.warning(
                        f"Resampling to target_vocos_sr ({self.target_vocos_sr}Hz) for {file_path} failed: {e}. Mel length may be inaccurate. Will attempt to use audio with sample rate {sr_orig}Hz.")

            num_samples_for_mel = waveform_for_mel_calc_cpu_batched.shape[-1]
            if self.config['vocos']['mel'].get('center', True):
                mel_length_actual = num_samples_for_mel // self.mel_hop_length + 1
            else:
                if num_samples_for_mel < self.mel_n_fft:
                    mel_length_actual = 1
                else:
                    mel_length_actual = (num_samples_for_mel - self.mel_n_fft) // self.mel_hop_length + 1
            mel_length_actual = max(1, mel_length_actual)

            waveform_main_sr_cpu_batched = waveform_orig_cpu_batched
            if sr_orig != self.sample_rate:
                try:
                    waveform_main_sr_cpu_batched = torchaudio.functional.resample(
                        waveform_orig_cpu_batched, sr_orig, self.sample_rate
                    )
                except Exception as e:
                    logger.warning(
                        f"Resampling to main sample rate ({self.sample_rate}Hz) for {file_path} failed: {e}. Will attempt to use audio with sample rate {sr_orig}Hz for subsequent processing.")

            target_audio_samples = self.config['audio']['max_length']
            current_len_main_sr = waveform_main_sr_cpu_batched.size(1)

            if current_len_main_sr > target_audio_samples:
                waveform_processed_cpu_batched_dim = waveform_main_sr_cpu_batched[:, :target_audio_samples]
            else:
                pad_length = target_audio_samples - current_len_main_sr
                waveform_processed_cpu_batched_dim = F.pad(waveform_main_sr_cpu_batched, (0, pad_length))

            waveform_processed_final_cpu = waveform_processed_cpu_batched_dim.squeeze(0)

            return {
                'waveform': waveform_processed_final_cpu,
                'file_path': str(file_path),
                'mel_lengths': torch.tensor(mel_length_actual, dtype=torch.long, device=torch.device('cpu'))
            }
        except Exception as e:
            failed_path_info = self.file_paths[idx] if idx < len(self.file_paths) else "Unknown path"
            logger.error(
                f"Failed to get sample (idx={idx}, path='{failed_path_info}'): {str(e)}\n{traceback.format_exc()}")
            return {
                'waveform': self.default_features['waveform'].clone(),
                'file_path': self.default_features['file_path'],
                'mel_lengths': self.default_features['mel_lengths'].clone()
            }
