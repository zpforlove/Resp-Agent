import argparse
import faulthandler
import logging
import math
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

import deepspeed
import matplotlib
import torch
import torch.nn.functional as F
import torchaudio
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from vocos import Vocos

import wandb
from beats.Tokenizers import TokenizersConfig, Tokenizers
from beats.BEATs import BEATs, BEATsConfig
from data_loader import FusionDataset
from models import MelSpectrogramExtractor, FlowMatchingModel
from utils import load_config, peak_norm

# Global variable declaration
global logger


def setup_logger_ds(rank=-1, config=None, level_override=None):
    current_logger = logging.getLogger("train_cfm_script")
    if current_logger.hasHandlers():
        current_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    log_format = f'%(asctime)s - RANK {rank} - %(module)s - %(levelname)s - %(message)s' if rank != -1 \
        else '%(asctime)s - %(module)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    current_logger.addHandler(handler)
    log_level_str = 'INFO'
    if config and 'logging' in config and \
            isinstance(config['logging'], dict) and 'console' in config['logging'] and \
            isinstance(config['logging']['console'], dict) and 'level' in config['logging']['console']:
        log_level_str = config['logging']['console']['level'].upper()
    if level_override:
        log_level_str = level_override.upper()
    log_level_resolved = getattr(logging, log_level_str, logging.INFO)
    current_logger.setLevel(log_level_resolved if rank <= 0 else logging.WARNING)
    current_logger.propagate = False
    return current_logger


class FlowMatchingTrainer:
    def __init__(self, config_dict, cmd_args):
        self.cmd_args = cmd_args
        self.config = config_dict
        self._is_shutdown = False
        global logger
        log_level_from_config = self.config.get('logging', {}).get('console', {}).get('level', 'INFO')
        logger = setup_logger_ds(self.cmd_args.local_rank, self.config, level_override=log_level_from_config)
        logger.info(f"Rank {self.cmd_args.local_rank}: Starting Flow Matching Trainer initialization...")

        self.mel_extractor = None
        self.flow_model = None
        self.pytorch_optimizer = None
        self.optimizer = None
        self.model_engine = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.wandb_initialized = False
        self.device = None

        # Declare BEATs tokenizer and feature extractor
        self.beats_tokenizer = None
        self.beats_feature_extractor = None

        self._setup_device()
        logger.info(f"Rank {self.cmd_args.local_rank}: Device setup completed. Using device: {self.device}")

        # Set up external model environment
        self._setup_external_models()

        self.setup_models()
        logger.info(f"Rank {self.cmd_args.local_rank}: Models, optimizer, and DeepSpeed engine initialized.")
        if self.cmd_args.local_rank <= 0:
            self.setup_wandb()
        logger.info(f"Rank {self.cmd_args.local_rank}: WandB setup completed.")
        self._init_data_loaders()
        logger.info(f"Rank {self.cmd_args.local_rank}: Dataloader initialized.")
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / 'flow'
        if self.cmd_args.local_rank <= 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rank {self.cmd_args.local_rank}: Checkpoint directory: {self.checkpoint_dir}")
        self.global_step = 0
        self.best_val_flow_loss = float('inf')
        self.current_epoch = 0

    def _setup_device(self):
        if self.model_engine:
            self.device = self.model_engine.device
        elif torch.cuda.is_available() and self.cmd_args.local_rank != -1:
            self.device = torch.device(f'cuda:{self.cmd_args.local_rank}')
        elif torch.cuda.is_available():
            gpu_id_to_use = self.config.get('device', {}).get('device_ids', [0])[0]
            self.device = torch.device(f'cuda:{gpu_id_to_use}')
        else:
            self.device = torch.device('cpu')
        logger.info(f"Rank {self.cmd_args.local_rank}: Device determined: {self.device}")

    def _setup_external_models(self):
        """Loads the Tokenizer for extracting BEATs Tokens and the Feature Extractor for extracting timbre."""
        try:
            # 1. Load BEATs Tokenizer
            beats_tokenizer_path = self.config['paths']['beats_tokenizer']
            if not os.path.exists(beats_tokenizer_path):
                raise FileNotFoundError(f"BEATs Tokenizer checkpoint not found at: {beats_tokenizer_path}")

            logger.info(f"Rank {self.cmd_args.local_rank}: Loading BEATs Tokenizer from {beats_tokenizer_path}...")
            tokenizer_checkpoint = torch.load(beats_tokenizer_path, map_location='cpu')
            tokenizer_cfg = TokenizersConfig(tokenizer_checkpoint['cfg'])
            self.beats_tokenizer = Tokenizers(tokenizer_cfg)
            self.beats_tokenizer.load_state_dict(tokenizer_checkpoint['model'])
            self.beats_tokenizer.to(self.device).eval()
            logger.info(
                f"Rank {self.cmd_args.local_rank}: BEATs Tokenizer loaded and moved to {self.device} in eval mode.")

            # 2. Load BEATs Feature Extractor
            beats_extractor_path = self.config['paths']['beats_feature_extractor_checkpoint']
            if not os.path.exists(beats_extractor_path):
                raise FileNotFoundError(f"BEATs Feature Extractor checkpoint not found at: {beats_extractor_path}")

            logger.info(
                f"Rank {self.cmd_args.local_rank}: Loading BEATs Feature Extractor from {beats_extractor_path}...")
            extractor_checkpoint = torch.load(beats_extractor_path, map_location='cpu')
            extractor_cfg = BEATsConfig(extractor_checkpoint['cfg'])
            self.beats_feature_extractor = BEATs(extractor_cfg)
            self.beats_feature_extractor.load_state_dict(extractor_checkpoint['model'])
            self.beats_feature_extractor.to(self.device).eval()
            logger.info(
                f"Rank {self.cmd_args.local_rank}: BEATs Feature Extractor loaded and moved to {self.device} in eval mode.")

        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: External models setup failed: {e}\n{traceback.format_exc()}")
            raise

    def setup_wandb(self):
        if self.cmd_args.local_rank != 0:
            self.wandb_initialized = False
            return
        try:
            if self.config['logging']['wandb']['enabled']:
                project = self.config['logging']['wandb']['project']
                name_prefix = self.config['logging']['wandb']['name']
                exp_name_suffix = self.config.get('meta', {}).get('exp_name', 'CFM')
                name = f"{name_prefix}_{exp_name_suffix}"
                wandb.init(project=project, name=name, config={**self.config, **vars(self.cmd_args)})
                logger.info(f"Rank 0: WandB initialized - Project: {project}, Run: {name}")
                self.wandb_initialized = True
            else:
                logger.info("Rank 0: WandB logging disabled in config.")
                self.wandb_initialized = False
        except Exception as e:
            logger.error(f"Rank 0: WandB setup failed: {str(e)}")
            self.wandb_initialized = False

    def setup_models(self):
        logger.info(f"Rank {self.cmd_args.local_rank}: Initializing models and optimizer...")
        try:
            self.mel_extractor = MelSpectrogramExtractor(self.config, target_device=self.device)
            logger.info(f"Rank {self.cmd_args.local_rank}: Mel Spectrogram Extractor Initialized.")
            self.flow_model = FlowMatchingModel(self.config)
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Flow Matching Model (with internal token_embedding) Instantiated.")
            self.pytorch_optimizer = torch.optim.AdamW(
                self.flow_model.parameters(),
                lr=float(self.config['hyperparameters']['flow']['optimizer']['scheduler']['max_lr']),
                weight_decay=float(self.config['hyperparameters']['flow']['optimizer']['weight_decay']),
                betas=eval(str(self.config['hyperparameters']['flow']['optimizer']['betas']))
            )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: PyTorch AdamW optimizer created for FlowMatchingModel parameters.")
            logger.info(f"Rank {self.cmd_args.local_rank}: Initializing DeepSpeed engine...")
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                args=self.cmd_args, model=self.flow_model, optimizer=self.pytorch_optimizer
            )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: DeepSpeed engine initialized. Model is on device: {self.model_engine.device}")
            self.device = self.model_engine.device

            # Move external models to the finally determined device
            if self.beats_tokenizer:
                self.beats_tokenizer.to(self.device)
            if self.beats_feature_extractor:
                self.beats_feature_extractor.to(self.device)

            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
            logger.info(f"Rank {self.cmd_args.local_rank}: Vocos loaded and moved to {self.device}.")
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Model or DeepSpeed initialization failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def setup_lr_scheduler(self):
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Optimizer or Train Loader not initialized. Cannot create LR scheduler.")
            return
        logger.info(f"Rank {self.cmd_args.local_rank}: Setting up LR scheduler...")
        scheduler_config = self.config['hyperparameters']['flow']['optimizer']['scheduler']
        num_epochs = self.config['hyperparameters']['flow']['num_epochs']
        gradient_accumulation_steps = self.model_engine.gradient_accumulation_steps()
        num_optimizer_steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        total_steps = num_optimizer_steps_per_epoch * num_epochs
        if total_steps <= 0:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Calculated total_steps ({total_steps}) is invalid. LR scheduler not created.")
            return
        logger.info(f"Rank {self.cmd_args.local_rank}: Total steps for LR scheduler: {total_steps}")
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.pytorch_optimizer, max_lr=float(scheduler_config['max_lr']),
            total_steps=total_steps, pct_start=float(scheduler_config['pct_start']),
            div_factor=float(scheduler_config['div_factor']),
            final_div_factor=float(scheduler_config['final_div_factor'])
        )
        logger.info(f"Rank {self.cmd_args.local_rank}: OneCycleLR scheduler created.")

    def _init_data_loaders(self):
        try:
            logger.info(f"Rank {self.cmd_args.local_rank}: Initializing data loaders...")
            if self.device is None:
                logger.warning(
                    f"Rank {self.cmd_args.local_rank}: self.device is None before data loader init. Attempting to use model_engine.device.")
                if self.model_engine:
                    self.device = self.model_engine.device
                else:
                    self.device = torch.device(
                        f"cuda:{self.cmd_args.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
                    logger.error(
                        f"Rank {self.cmd_args.local_rank}: model_engine not available, fallback device: {self.device}.")
            is_distributed = torch.distributed.is_initialized() and self.cmd_args.local_rank != -1
            current_rank = self.cmd_args.local_rank if is_distributed else 0
            world_size = torch.distributed.get_world_size() if is_distributed else 1
            logger.info(
                f"Rank {current_rank}/{world_size} (is_distributed={is_distributed}): Setting up data loaders...")
            train_dir = Path(self.config['data']['train_root'])
            val_dir = Path(self.config['data']['val_root'])
            if not train_dir.exists(): raise ValueError(f"Training data directory does not exist: {train_dir}")
            if not val_dir.exists(): raise ValueError(f"Validation data directory does not exist: {val_dir}")
            target_device_for_dataset_str = str(self.device)
            logger.info(
                f"Rank {current_rank}: Instantiating datasets with target_device '{target_device_for_dataset_str}'")
            train_dataset = FusionDataset(root_dir=train_dir, config=self.config,
                                          target_device=target_device_for_dataset_str)
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=current_rank,
                                               shuffle=True) if is_distributed else None
            batch_size_per_gpu = self.model_engine.train_micro_batch_size_per_gpu()
            logger.info(
                f"Rank {current_rank}: train_micro_batch_size_per_gpu from DeepSpeed engine: {batch_size_per_gpu}")
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler,
                shuffle=(train_sampler is None), drop_last=True,
                num_workers=self.config.get('data', {}).get('num_workers', 4),
                pin_memory=self.config.get('data', {}).get('pin_memory', True)
            )
            logger.info(f"Rank {current_rank}: Training DataLoader created. Num batches: {len(self.train_loader)}")
            val_dataset = FusionDataset(root_dir=val_dir, config=self.config,
                                        target_device=target_device_for_dataset_str)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=current_rank,
                                             shuffle=False) if is_distributed else None
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, shuffle=False,
                num_workers=self.config.get('data', {}).get('num_workers', 4),
                pin_memory=self.config.get('data', {}).get('pin_memory', True)
            )
            logger.info(f"Rank {current_rank}: Validation DataLoader created. Num batches: {len(self.val_loader)}")
            self.setup_lr_scheduler()
        except Exception as e:
            rank_for_log = self.cmd_args.local_rank
            logger.error(f"Rank {rank_for_log}: Failed to set up data loaders: {str(e)}\n{traceback.format_exc()}")
            if hasattr(self, 'cleanup'): self.cleanup()
            raise

    def _get_beats_tokens(self, waveforms):
        """Extracts BEATs Token indices from waveforms using BEATs Tokenizer and corrects their output shape"""
        with torch.no_grad():
            # First, get the batch size
            batch_size = waveforms.shape[0]

            # Extract BEATs labels. This will return a 1D, concatenated tensor
            flat_labels = self.beats_tokenizer.extract_labels(waveforms)

            # If the batch size is 1, the tokenizer might return a 1D tensor directly, we need to add a batch dimension
            if batch_size == 1 and flat_labels.dim() == 1:
                return flat_labels.unsqueeze(0)  # [seq_len] -> [1, seq_len]

            # For batch sizes greater than 1, we reshape the concatenated 1D tensor to [batch_size, seq_len]
            # Using .view(batch_size, -1) can automatically calculate the sequence length for each sample
            if batch_size > 1:
                reshaped_labels = flat_labels.view(batch_size, -1)
                return reshaped_labels

            # As a fallback, if something unexpected happens (e.g., batch_size=0)
            return flat_labels

    def prepare_batch(self, batch):
        """
        Prepares a single batch of data for the flow matching model.
        This function generates for each sample:
        1. The target mel-spectrogram (full_mel_specs).
        2. A conditional embedding dictionary (cond_embed_dict), containing:
           - 'fused_embed': An embedding that fuses content (BEATs token) and timbre (BEATs feature) information.
           - 'ref_mel_for_cond': A reference mel-spectrogram fragment to guide generation.
        """
        try:
            waveforms = batch['waveform'].to(self.device, non_blocking=True)
            full_mel_specs = self.mel_extractor(waveforms).to(self.device, non_blocking=True)
            B, n_mels, T_full_mel = full_mel_specs.shape

            # 1. Prepare reference mel-spectrogram condition (Ref Mel)
            ref_mel_parts = []
            actual_ref_lengths = []
            for i in range(B):
                current_mel_len = batch['mel_lengths'][i].item()
                max_ref_len_abs = max(1, int(current_mel_len * 0.3))
                ref_len = torch.randint(1, max_ref_len_abs + 1, (1,)).item() if max_ref_len_abs > 0 else 0
                actual_ref_lengths.append(ref_len)
                ref_part = full_mel_specs[i, :, :ref_len] if ref_len > 0 else torch.empty((n_mels, 0),
                                                                                          device=self.device,
                                                                                          dtype=full_mel_specs.dtype)
                ref_mel_parts.append(ref_part)

            ref_mel_for_cond_list = []
            for ref_part_item in ref_mel_parts:
                pad_amount = T_full_mel - ref_part_item.shape[1]
                if pad_amount > 0:
                    ref_part_padded = F.pad(ref_part_item, (0, pad_amount), value=0)
                else:
                    # If the reference part is longer (unlikely, but as a safeguard), truncate it
                    ref_part_padded = ref_part_item[:, :T_full_mel]
                ref_mel_for_cond_list.append(ref_part_padded)
            ref_mel_for_cond = torch.stack(ref_mel_for_cond_list)

            with torch.no_grad():
                # 2. Extract content embedding (Content Embedding from BEATs Tokens)
                beats_indices = self._get_beats_tokens(waveforms)  # [B, T_token]
                token_embed = self.model_engine.module.token_embedding(beats_indices).permute(0, 2,
                                                                                              1)  # -> [B, D_emb, T_token]
                token_embed_resampled = F.interpolate(token_embed, size=T_full_mel, mode='linear', align_corners=False)

                # 3. Extract timbre embedding (Timbre Embedding from BEATs Features)
                # a) Extract raw BEATs features
                beats_features, _ = self.beats_feature_extractor.extract_features(waveforms)  # (B, T_feat, D_feat)
                # b) Global average pooling to get the timbre vector
                timbre_embed = beats_features.mean(dim=1)  # -> (B, D_feat)
                # c) Expand to the same time length as the mel-spectrogram
                timbre_embed_expanded = timbre_embed.unsqueeze(-1).expand(-1, -1,
                                                                          T_full_mel)  # -> (B, D_feat, T_full_mel)

            # 4. Fuse content and timbre embeddings
            fused_embed = torch.cat([token_embed_resampled, timbre_embed_expanded],
                                    dim=1)  # [B, D_emb + D_feat, T_full_mel]

            cond_embed_dict = {
                'fused_embed': fused_embed,
                'ref_mel_for_cond': ref_mel_for_cond
            }

            return {
                'full_mel_specs': full_mel_specs,
                'cond_embed_dict': cond_embed_dict,
                'ref_mel_parts_for_viz': ref_mel_parts,
                'target_mel_parts_for_viz': [full_mel_specs[i, :, actual_ref_lengths[i]:] for i in range(B)],
                'actual_ref_lengths': actual_ref_lengths,
                'full_mel_lengths_for_viz': batch['mel_lengths']
            }

        except Exception as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Batch preparation failed: {e}\n{traceback.format_exc()}")
            raise

    def train_step(self, batch):
        self.model_engine.train()
        data = self.prepare_batch(batch)
        x = data['full_mel_specs']
        cond_embed_dict = data['cond_embed_dict']

        B = x.size(0)
        t = torch.rand(B, device=x.device)

        flow_loss = self.model_engine(x=x, t=t, cond_embed_dict=cond_embed_dict)

        is_loss_valid = True
        if not torch.isfinite(flow_loss):
            logger.error(
                f"Rank {self.cmd_args.local_rank}: NaN/Inf detected in training loss: {flow_loss.item()}. Skipping step.")
            flow_loss_item = 0.0
            is_loss_valid = False
            if self.optimizer: self.optimizer.zero_grad()
        else:
            flow_loss_item = flow_loss.item()

        if is_loss_valid:
            # Use the original loss for backpropagation
            self.model_engine.backward(flow_loss)
            self.model_engine.step()

        if self.lr_scheduler and is_loss_valid and self.model_engine.is_gradient_accumulation_boundary():
            self.lr_scheduler.step()

        return {'flow_loss': flow_loss_item, 'is_valid_step': is_loss_valid}

    def train(self):
        logger.info(f"Rank {self.cmd_args.local_rank}: Starting Conditional Flow Matching training...")
        num_epochs = self.config['hyperparameters']['flow']['num_epochs']
        is_main_process = self.cmd_args.local_rank <= 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()  # All ranks set the model to training mode
            if isinstance(self.train_loader.sampler, DistributedSampler):  # All ranks set the sampler's epoch
                self.train_loader.sampler.set_epoch(epoch)

            epoch_flow_loss_sum = 0.0
            num_valid_batches_epoch = 0

            # Progress bar for the training loop is only displayed on Rank 0
            pbar_desc = f"Epoch {epoch + 1}/{num_epochs} [Training...]"
            pbar = tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc=pbar_desc, ncols=150, disable=not is_main_process
            )

            for step, batch in pbar:
                # train_step should be executed by all ranks, it contains model forward/backward and optimizer steps
                loss_info = self.train_step(batch)
                if loss_info['is_valid_step']:
                    epoch_flow_loss_sum += loss_info['flow_loss']
                    num_valid_batches_epoch += 1
                self.global_step += 1

                if is_main_process:  # Rank 0 updates the progress bar and WandB step logs
                    current_lr = self.pytorch_optimizer.param_groups[0]['lr'] if self.pytorch_optimizer else float(
                        'nan')
                    avg_epoch_loss_so_far = epoch_flow_loss_sum / num_valid_batches_epoch if num_valid_batches_epoch > 0 else 0.0
                    pbar.set_postfix({
                        'flow_loss': f"{loss_info['flow_loss']:.4f}",
                        'avg_flow': f"{avg_epoch_loss_so_far:.4f}",
                        'lr': f"{current_lr:.2e}"
                    })
                    if self.wandb_initialized and loss_info['is_valid_step']:
                        try:
                            wandb.log(
                                {'train/step_flow_loss': loss_info['flow_loss'], 'train/learning_rate': current_lr},
                                step=self.global_step)
                        except Exception as e:
                            logger.error(
                                f"Rank 0: WandB step logging error at global_step {self.global_step}: {e}\n{traceback.format_exc()}")

            # Synchronize training loss statistics (executed by all ranks)
            epoch_stats_tensor = torch.tensor([epoch_flow_loss_sum, float(num_valid_batches_epoch)],
                                              dtype=torch.float32, device=self.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(epoch_stats_tensor, op=torch.distributed.ReduceOp.SUM)

            global_epoch_flow_loss_sum = epoch_stats_tensor[0].item()
            global_num_valid_batches_epoch = epoch_stats_tensor[1].item()
            avg_epoch_flow_loss = global_epoch_flow_loss_sum / global_num_valid_batches_epoch if global_num_valid_batches_epoch > 0 else 0.0

            if is_main_process:  # Rank 0 logs the aggregated training loss
                logger.info(
                    f"Rank 0: Epoch {epoch + 1} completed. Average Training Flow Loss (Global): {avg_epoch_flow_loss:.4f}")

            # --- Perform Validation ---
            # The validate() function has an internal all_reduce, so it must be called by all ranks
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Epoch {epoch + 1} training completed. Starting validation...")
            val_metrics = self.validate(epoch)  # ALL RANKS CALL VALIDATE
            val_flow_loss = val_metrics[
                'val_flow_loss']  # val_flow_loss is consistent across all ranks (due to all_reduce inside validate)

            # --- Rank 0 makes decisions and logs ---
            should_save_this_epoch = torch.tensor([0], dtype=torch.int, device=self.device)
            if is_main_process:
                logger.info(f"Rank 0: Epoch {epoch + 1} Validation Flow Loss (Global): {val_flow_loss:.4f}")
                if val_flow_loss < self.best_val_flow_loss:
                    self.best_val_flow_loss = val_flow_loss  # Update Rank 0's best loss record
                    logger.info(
                        f"Rank 0: New best validation flow loss: {self.best_val_flow_loss:.4f}. Marking for checkpoint save.")
                    should_save_this_epoch[0] = 1

                if self.wandb_initialized:  # Rank 0 logs W&B epoch logs
                    try:
                        wandb.log({'epoch/epoch_num': epoch + 1,
                                   'epoch/train_avg_flow_loss': avg_epoch_flow_loss,  # Global training loss
                                   'epoch/val_avg_flow_loss': val_flow_loss,  # Global validation loss
                                   'epoch/best_val_flow_loss': self.best_val_flow_loss},  # Rank 0's best record
                                  step=self.global_step)
                    except Exception as e:
                        logger.error(
                            f"Rank 0: WandB epoch logging error for epoch {epoch + 1}: {e}\n{traceback.format_exc()}")

            # --- Checkpoint Saving ---
            # Broadcast the save decision from Rank 0 to all ranks
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(should_save_this_epoch, src=0)

            if should_save_this_epoch.item() == 1:
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Received signal to save checkpoint for epoch {epoch + 1}.")
                # save_checkpoint() has an internal barrier, and only rank 0 writes, so all ranks need to call it
                # Rank 0 needs the actual tag and client_state, other ranks can pass null or default values
                tag_to_save = ""
                client_state_to_save = {}
                if is_main_process:  # Rank 0 prepares the actual tag and client_state
                    tag_to_save = f"best_ep{epoch + 1}_val_loss_{self.best_val_flow_loss:.4f}_step{self.global_step}"
                    client_state_to_save = {'epoch': epoch + 1, 'global_step': self.global_step,
                                            'best_val_metric': self.best_val_flow_loss,
                                            'config_snapshot': self.config}
                    logger.info(f"Rank 0: Checkpoint tag: {tag_to_save}")

                # All ranks call save_checkpoint
                self.save_checkpoint(tag=tag_to_save, client_state=client_state_to_save)

            # Barrier at the end of the epoch to ensure all processes synchronize before starting the next epoch or finishing training
            if torch.distributed.is_initialized():
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Reached end of epoch {epoch + 1}. Synchronizing all processes.")
                torch.distributed.barrier()
                logger.info(f"Rank {self.cmd_args.local_rank}: Passed end of epoch {epoch + 1} barrier.")

        if is_main_process:
            logger.info(f"Training completed! Best validation flow loss (Rank 0): {self.best_val_flow_loss:.4f}")

        # Final barrier after training finishes
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def validate_step(self, batch):
        self.model_engine.eval()
        with torch.no_grad():
            data = self.prepare_batch(batch)
            full_mel = data['full_mel_specs']
            cond_embed_dict_for_loss = data['cond_embed_dict']

            B, _, T_full_mel = full_mel.shape
            t = torch.rand(B, device=self.device)

            flow_loss = self.model_engine(x=full_mel, t=t, cond_embed_dict=cond_embed_dict_for_loss)
            flow_loss_item = flow_loss.item()

            if not math.isfinite(flow_loss_item):
                logger.warning(
                    f"Rank {self.cmd_args.local_rank}: Validation loss is NaN/Inf ({flow_loss_item}). Setting to Inf.")
                flow_loss_item = float('inf')

            viz_data = {}
            if self.cmd_args.local_rank <= 0:
                # Prepare data needed for visualization/audio generation
                viz_data = {
                    'ground_truth_full_mel_cpu': data['full_mel_specs'].cpu(),  # Complete ground truth mel-spectrogram
                    'cond_embed_dict_cpu': {k: v.cpu() for k, v in data['cond_embed_dict'].items()},
                    'ref_mel_parts_for_viz_cpu': [item.cpu() for item in data['ref_mel_parts_for_viz']],
                    'actual_ref_lengths_cpu': data['actual_ref_lengths'],
                    'full_mel_lengths_for_viz_cpu': data['full_mel_lengths_for_viz'].cpu()
                    # Length of the ground truth full mel-spectrogram
                }
            return {'val_flow_loss': flow_loss_item, 'batch_size': B, **viz_data}

    def validate(self, epoch):
        is_main_process = self.cmd_args.local_rank <= 0
        if is_main_process:
            logger.info(f"Rank 0: Starting validation for epoch {epoch + 1} (validate function)...")

        self.model_engine.eval()
        accumulated_val_flow_loss = torch.tensor(0.0, device=self.device)
        accumulated_val_samples = torch.tensor(0, device=self.device, dtype=torch.long)

        # To store the first batch's data on rank 0 for later visualization
        first_batch_data_for_rank0_viz = None

        if hasattr(self.val_loader, 'sampler') and isinstance(self.val_loader.sampler, DistributedSampler):
            self.val_loader.sampler.set_epoch(epoch)

        # Log the state for all processes upon entering the loop
        logger.info(
            f"Rank {self.cmd_args.local_rank}: Entering validation loop for epoch {epoch + 1}. Val loader size: {len(self.val_loader)}")

        val_pbar = tqdm.tqdm(
            enumerate(self.val_loader), total=len(self.val_loader),
            desc=f"Epoch {epoch + 1} [Validating...]", ncols=120, disable=not is_main_process
        )

        for step, batch in val_pbar:
            logger.debug(f"Rank {self.cmd_args.local_rank}: Epoch {epoch + 1}, Val Step {step}, Processing batch.")
            try:
                metrics = self.validate_step(batch)  # Executed by all ranks

                if metrics['val_flow_loss'] != float('inf'):
                    accumulated_val_flow_loss += metrics['val_flow_loss'] * metrics['batch_size']
                    accumulated_val_samples += metrics['batch_size']

                if is_main_process:
                    val_pbar.set_postfix({'val_flow_loss': f"{metrics['val_flow_loss']:.4f}"})
                    # Save the first valid batch's data from rank 0 for visualization after all_reduce
                    if step == 0 and first_batch_data_for_rank0_viz is None and metrics.get(
                            'ground_truth_full_mel_cpu') is not None:
                        # Ensure metrics contain all necessary data, and only take the first sample
                        # The viz_data returned by validate_step is a dictionary, ensure correct keys
                        if 'ground_truth_full_mel_cpu' in metrics and \
                                'cond_embed_dict_cpu' in metrics and \
                                'ref_mel_parts_for_viz_cpu' in metrics and \
                                'actual_ref_lengths_cpu' in metrics and \
                                'full_mel_lengths_for_viz_cpu' in metrics:
                            first_batch_data_for_rank0_viz = {
                                'ground_truth_full_mel_cpu': metrics['ground_truth_full_mel_cpu'][0:1].clone(),
                                # Take the first sample and clone
                                'cond_embed_dict_cpu': {k: v[0:1].clone() for k, v in
                                                        metrics['cond_embed_dict_cpu'].items()},
                                'ref_mel_parts_for_viz_cpu': [metrics['ref_mel_parts_for_viz_cpu'][0].clone()],
                                'actual_ref_lengths_cpu': [metrics['actual_ref_lengths_cpu'][0]],
                                # Usually a Python number, list of numbers
                                'full_mel_lengths_for_viz_cpu': metrics['full_mel_lengths_for_viz_cpu'][0:1].clone()
                            }
                            logger.info(f"Rank 0: Saved first batch data for visualization for epoch {epoch + 1}.")

            except Exception as e:
                logger.error(
                    f"Rank {self.cmd_args.local_rank}: A critical error occurred in validate_step (epoch {epoch + 1}, step {step}): {e}\n{traceback.format_exc()}")
                raise

        logger.info(
            f"Rank {self.cmd_args.local_rank}: Completed validation loop for epoch {epoch + 1}. Preparing for all_reduce.")
        logger.info(
            f"Rank {self.cmd_args.local_rank}: Before all_reduce: accumulated_val_flow_loss = {accumulated_val_flow_loss.item()}, accumulated_val_samples = {accumulated_val_samples.item()}")

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(accumulated_val_flow_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(accumulated_val_samples, op=torch.distributed.ReduceOp.SUM)
            logger.info(f"Rank {self.cmd_args.local_rank}: all_reduce completed.")

        logger.info(
            f"Rank {self.cmd_args.local_rank}: After all_reduce: global_accumulated_val_flow_loss = {accumulated_val_flow_loss.item()}, global_accumulated_val_samples = {accumulated_val_samples.item()}")

        avg_val_flow_loss_epoch = float('inf')
        if accumulated_val_samples.item() > 0:
            avg_val_flow_loss_epoch = accumulated_val_flow_loss.item() / accumulated_val_samples.item()
        elif is_main_process:
            logger.warning(
                f"Rank 0: Epoch {epoch + 1}: Global number of validation samples is 0. Average validation loss set to Inf.")

        # Move Rank 0 specific sampling and W&B media logging here
        if is_main_process and self.wandb_initialized and first_batch_data_for_rank0_viz is not None:
            if avg_val_flow_loss_epoch != float('inf'):  # Ensure validation had valid results before visualizing
                logger.info(f"Rank 0: Starting to generate W&B visualizations for epoch {epoch + 1}...")
                try:
                    wandb_epoch_log_data = {}  # In the original code, this variable was defined outside the loop for collecting media. Re-initialize or ensure its state is correct here.

                    output_dir_path_str = self.config.get("paths", {}).get("output_dir", "outputs_cfm")
                    val_output_dir_main = Path(output_dir_path_str) / f"val_epoch{epoch + 1}"
                    val_output_dir_main.mkdir(parents=True, exist_ok=True)

                    mel_mean = self.config['hyperparameters']['flow']['mel_mean']
                    mel_std = self.config['hyperparameters']['flow']['mel_std']

                    cond_dict_for_sample_dev = {
                        'fused_embed': first_batch_data_for_rank0_viz['cond_embed_dict_cpu']['fused_embed'].to(
                            self.device),
                        'ref_mel_for_cond': first_batch_data_for_rank0_viz['cond_embed_dict_cpu'][
                            'ref_mel_for_cond'].to(self.device)
                    }
                    first_sample_full_mel_len = first_batch_data_for_rank0_viz['full_mel_lengths_for_viz_cpu'][0].item()

                    generated_full_mel_dev = self.model_engine.module.sample(
                        cond_embed_dict=cond_dict_for_sample_dev,
                        target_duration_frames=first_sample_full_mel_len,
                        sway_sampling_coef=-1.0
                    )
                    generated_full_mel_cpu = generated_full_mel_dev.detach().cpu()

                    ref_mel_part_cpu_first = first_batch_data_for_rank0_viz['ref_mel_parts_for_viz_cpu'][0]
                    actual_ref_len_first = first_batch_data_for_rank0_viz['actual_ref_lengths_cpu'][0]

                    if actual_ref_len_first < first_sample_full_mel_len:
                        generated_mel_pred_part_cpu = generated_full_mel_cpu[0, :,
                        actual_ref_len_first:first_sample_full_mel_len]
                    else:
                        generated_mel_pred_part_cpu = torch.empty((self.config['vocos']['mel']['n_mels'], 0),
                                                                  dtype=generated_full_mel_cpu.dtype)

                    if ref_mel_part_cpu_first.shape[1] > 0:
                        reconstructed_mel_for_viz = torch.cat([ref_mel_part_cpu_first, generated_mel_pred_part_cpu],
                                                              dim=1)
                    else:
                        reconstructed_mel_for_viz = generated_mel_pred_part_cpu

                    ground_truth_mel_for_viz_first = first_batch_data_for_rank0_viz['ground_truth_full_mel_cpu'][0, :,
                    :reconstructed_mel_for_viz.shape[1]]

                    raw_true_mel_for_viz = ground_truth_mel_for_viz_first * mel_std + mel_mean
                    raw_reconstructed_mel_for_viz = reconstructed_mel_for_viz * mel_std + mel_mean

                    with matplotlib.rc_context(rc={'backend': 'Agg'}):
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True)
                        im_gt = ax1.imshow(raw_true_mel_for_viz.numpy(), aspect='auto', origin='lower',
                                           interpolation='none')
                        ax1.set_title(f"Val True Mel (Ep{epoch + 1} St0_viz, Len:{raw_true_mel_for_viz.shape[1]})")
                        fig.colorbar(im_gt, ax=ax1)

                        im_pred = ax2.imshow(raw_reconstructed_mel_for_viz.numpy(), aspect='auto', origin='lower',
                                             interpolation='none')
                        ax2.set_title(
                            f"Val Reconstructed Mel (RefLen:{actual_ref_len_first}, PredLen:{generated_mel_pred_part_cpu.shape[1]})")
                        fig.colorbar(im_pred, ax=ax2)
                        plt.tight_layout()
                        wandb_epoch_log_data[f'Val Media/Mel Comparison Ep{epoch + 1}'] = wandb.Image(fig)
                        plt.close(fig)

                    if hasattr(self, 'vocos') and self.vocos:
                        with torch.inference_mode():
                            audio_gt_dev = self.vocos.decode(raw_true_mel_for_viz.unsqueeze(0).to(self.device))
                            audio_reconstructed_dev = self.vocos.decode(
                                raw_reconstructed_mel_for_viz.unsqueeze(0).to(self.device))

                            audio_gt_cpu = peak_norm(audio_gt_dev.cpu())
                            audio_reconstructed_cpu = peak_norm(audio_reconstructed_dev.cpu())

                        sample_rate = self.config['vocos']['sample_rate']
                        true_audio_path = val_output_dir_main / f"epoch{epoch + 1}_step0_audio_GT_viz.wav"
                        reconstructed_audio_path = val_output_dir_main / f"epoch{epoch + 1}_step0_audio_RECONSTRUCTED_viz.wav"

                        torchaudio.save(str(true_audio_path), audio_gt_cpu, sample_rate, format="wav")
                        torchaudio.save(str(reconstructed_audio_path), audio_reconstructed_cpu, sample_rate,
                                        format="wav")

                        wandb_epoch_log_data[f'Val Media/Audio Real Ep{epoch + 1}'] = wandb.Audio(str(true_audio_path),
                                                                                                  sample_rate=sample_rate)
                        wandb_epoch_log_data[f'Val Media/Audio Reconstructed Ep{epoch + 1}'] = wandb.Audio(
                            str(reconstructed_audio_path), sample_rate=sample_rate)

                    if wandb_epoch_log_data:  # Ensure there is media data to log
                        wandb.log(wandb_epoch_log_data, step=self.global_step)
                    logger.info(f"Rank 0: W&B visualizations logged for epoch {epoch + 1}.")
                except Exception as e_viz:
                    logger.error(
                        f"Rank 0: An error occurred during W&B visualization/media logging (epoch {epoch + 1}): {e_viz}\n{traceback.format_exc()}")
            elif is_main_process and self.wandb_initialized and first_batch_data_for_rank0_viz is None:
                logger.warning(
                    f"Rank 0: Failed to get the first batch data for W&B visualization for epoch {epoch + 1}.")

        logger.info(
            f"Rank {self.cmd_args.local_rank}: Exiting validate function for epoch {epoch + 1}. Avg loss: {avg_val_flow_loss_epoch}")
        return {'val_flow_loss': avg_val_flow_loss_epoch}

    def save_checkpoint(self, tag: str, client_state: dict):
        # Added barrier: ensure all processes synchronize before rank 0 starts saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.cmd_args.local_rank <= 0:
            checkpoint_file_path = self.checkpoint_dir / f"{tag}.pt"
            logger.info(f"Rank 0: Preparing to save checkpoint to {checkpoint_file_path}")
            model_state_dict = self.model_engine.module.state_dict()
            content_to_save = {
                'model_state_dict': model_state_dict,
                'pytorch_optimizer_state_dict': self.pytorch_optimizer.state_dict() if self.pytorch_optimizer else None,
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            }
            if client_state: content_to_save.update(client_state)
            try:
                torch.save(content_to_save, checkpoint_file_path)
                logger.info(f"Rank 0: Checkpoint successfully saved to {checkpoint_file_path}")
            except Exception as e:
                logger.error(f"Rank 0: Failed to save checkpoint {checkpoint_file_path}: {e}\n{traceback.format_exc()}")
                raise

        # Retain the original trailing barrier: ensure all processes wait for rank 0 to finish saving before continuing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(self, checkpoint_path_str: str):
        checkpoint_path = Path(checkpoint_path_str)
        if not checkpoint_path.exists():
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Checkpoint file not found at {checkpoint_path}. Cannot load.")
            return False
        logger.info(f"Rank {self.cmd_args.local_rank}: Attempting to load checkpoint from {checkpoint_path}...")
        try:
            map_location = {
                'cuda:%d' % 0: 'cuda:%d' % self.cmd_args.local_rank} if self.cmd_args.local_rank != -1 else self.device
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            if 'model_state_dict' in checkpoint:
                self.model_engine.module.load_state_dict(checkpoint['model_state_dict'])
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Loaded model_state_dict (FlowMatchingModel with internal embedding).")
            if self.pytorch_optimizer and 'pytorch_optimizer_state_dict' in checkpoint:
                self.pytorch_optimizer.load_state_dict(checkpoint['pytorch_optimizer_state_dict'])
                logger.info(f"Rank {self.cmd_args.local_rank}: Loaded pytorch_optimizer_state_dict.")
                for state in self.pytorch_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(self.device)
            if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                logger.info(f"Rank {self.cmd_args.local_rank}: Loaded lr_scheduler_state_dict.")
            self.global_step = checkpoint.get('global_step', 0)
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_flow_loss = checkpoint.get('best_val_metric', float('inf'))
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Checkpoint loaded. Resuming from epoch {self.current_epoch}, step {self.global_step}, best_val_loss {self.best_val_flow_loss:.4f}")
            return True
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to load checkpoint from {checkpoint_path}: {str(e)}\n{traceback.format_exc()}")
            return False

    def cleanup(self):
        rank = self.cmd_args.local_rank
        if rank <= 0 and self.wandb_initialized and wandb.run:
            try:
                wandb.finish()
                logger.info(f"Rank {rank}: WandB run finished.")
            except Exception as e_wb_finish:
                logger.error(f"Rank {rank}: Error finishing WandB run: {e_wb_finish}")


def main():
    global logger
    os.environ['MPLBACKEND'] = 'Agg'
    matplotlib.use('Agg', force=True)
    if sys.platform != 'win32':
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method not in ['spawn', 'forkserver']:
            try:
                multiprocessing.set_start_method('spawn', force=True)
                if int(os.getenv('LOCAL_RANK', -1)) <= 0: print("INFO: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e_mp_set:
                if int(os.getenv('LOCAL_RANK', -1)) <= 0: print(
                    f"WARNING: Failed to set 'spawn' start method: {e_mp_set}")
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    faulthandler.enable()

    parser = argparse.ArgumentParser(description="Distributed Flow Matching Training Script (CFM)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the main configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher.')
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    if cmd_args.local_rank == -1:
        env_local_rank = os.getenv('LOCAL_RANK')
        if env_local_rank is not None:
            try:
                cmd_args.local_rank = int(env_local_rank)
            except ValueError:
                pass
    try:
        config_obj = load_config(cmd_args.config)
    except Exception as e_cfg_load:
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"FATAL: Failed to load main config '{cmd_args.config}': {e_cfg_load}")
        sys.exit(1)

    log_level_from_config = config_obj.get('logging', {}).get('console', {}).get('level', 'INFO')
    logger = setup_logger_ds(cmd_args.local_rank, config_obj, level_override=log_level_from_config)
    deepspeed.runtime.utils.set_random_seed(config_obj.get('meta', {}).get('seed', 666))
    logger.info(f"Rank {cmd_args.local_rank}: Global random seed set via DeepSpeed.")

    if cmd_args.local_rank != -1:
        logger.info(f"Rank {cmd_args.local_rank}: Initializing DeepSpeed distributed environment...")
        deepspeed.init_distributed()
        logger.info(
            f"Rank {cmd_args.local_rank}: Distributed environment initialized. World size: {torch.distributed.get_world_size()}")
    else:
        logger.info(f"Rank {cmd_args.local_rank}: Not a distributed run (or local_rank is -1).")

    trainer_instance = None
    try:
        trainer_instance = FlowMatchingTrainer(config_obj, cmd_args)
        if cmd_args.resume:
            trainer_instance.load_checkpoint(cmd_args.resume)
            if torch.distributed.is_initialized(): torch.distributed.barrier()
        trainer_instance.train()
    except KeyboardInterrupt:
        if logger: logger.warning(f"Rank {cmd_args.local_rank}: Training interrupted by user (KeyboardInterrupt).")
    except Exception as e_main_loop:
        if logger:
            logger.error(
                f"Rank {cmd_args.local_rank}: Unhandled exception in main training loop: {e_main_loop}\n{traceback.format_exc()}")
        else:
            print(f"Rank {cmd_args.local_rank} MAIN_LOOP_ERROR: {e_main_loop}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        if trainer_instance is not None and hasattr(trainer_instance, 'cleanup'):
            trainer_instance.cleanup()
        if hasattr(cmd_args, 'local_rank') and cmd_args.local_rank != -1 and torch.distributed.is_initialized():
            logger.info(f"Rank {cmd_args.local_rank}: Reached end of main, waiting at final barrier.")
            torch.distributed.barrier()
            logger.info(f"Rank {cmd_args.local_rank}: Passed final barrier.")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Rank {cmd_args.local_rank}: Main function finished.")


if __name__ == "__main__":
    main()
