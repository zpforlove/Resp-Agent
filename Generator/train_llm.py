import argparse
import faulthandler
import logging
import json
import math
import os
import sys
import traceback
from pathlib import Path

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from beats.Tokenizers import TokenizersConfig, Tokenizers
from beats.BEATs import BEATs, BEATsConfig
from data_loader import FusionDataset
from utils import load_config

# Global variable declaration
global trainer_global_object

# --- Logger Setup ---
global logger


def setup_logger(rank: int = -1):
    """Set up a global logger based on the process rank"""
    current_logger = logging.getLogger(__name__)
    if current_logger.hasHandlers():
        current_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    # Use f-string to inject rank to avoid conflicts with logging's %(...) placeholders
    log_format = (
        f'%(asctime)s - RANK {rank} - %(name)s - %(levelname)s - %(message)s'
        if rank != -1 else
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    current_logger.addHandler(handler)  # ← This line is essential!

    # rank=-1 or 0 logs INFO, other ranks log WARNING to reduce multi-process spam
    current_logger.setLevel(logging.INFO if rank in (-1, 0) else logging.WARNING)
    current_logger.propagate = False

    global logger
    logger = current_logger
    return logger


class LLMTrainer:
    def __init__(self, config_path, cmd_args):
        """
        Initializes the LLM trainer, adapted for DeepSpeed and the new FusionDataset data flow.

        Args:
            config_path: Path to the configuration file.
            cmd_args: Command line arguments, including deepspeed configuration.
        """
        self.cmd_args = cmd_args
        self.config = load_config(config_path)
        global logger
        if logger is None:
            logger = setup_logger(self.cmd_args.local_rank)

        print(f"Rank {self.cmd_args.local_rank}: Starting trainer initialization...")

        # Initialize components
        self.llm_tokenizer = None
        self.model_engine = None
        self.optimizer = None
        self.pytorch_optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.device = None
        self.beats_feature_extractor = None  # For extracting timbre features

        # Metadata loading
        self.ontology_map = None
        self.train_segments_df = None
        self.eval_segments_df = None

        self._setup_environment()
        if self.cmd_args.local_rank <= 0:
            self._setup_wandb()
        self._setup_model_and_deepspeed()

        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / 'llm'
        if self.cmd_args.local_rank <= 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoint directory created/confirmed: {self.checkpoint_dir}")

        print(f"Rank {self.cmd_args.local_rank}: LLM trainer initialized successfully")

    def _load_metadata(self):
        """
        Loads the audio_descriptions.jsonl file to create a mapping from audio filenames to disease diagnoses.
        """
        print("Loading audio_descriptions.jsonl metadata...")
        self.filename_to_disease = {}
        jsonl_path = "audio_descriptions.jsonl"

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        filename = record.get("audio_filename")
                        disease = record.get("disease")
                        if filename and disease:
                            self.filename_to_disease[filename] = disease
                        else:
                            print(f"[WARNING] Skipping malformed line: {line.strip()}")
                    except json.JSONDecodeError:
                        print(f"[WARNING] Could not parse JSON line: {line.strip()}")
            print(f"Successfully loaded metadata file: {jsonl_path}")
        except FileNotFoundError:
            print(f"[ERROR] Metadata file not found: {jsonl_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to load metadata file: {jsonl_path}, error: {e}", file=sys.stderr)
            sys.exit(1)

        if not self.filename_to_disease:
            print(
                "[ERROR] Failed to load any diagnosis data, mapping is empty. Please check the path and content of audio_descriptions.jsonl.",
                file=sys.stderr)
            sys.exit(1)

        print(f"Metadata loading complete. Found {len(self.filename_to_disease)} audio file diagnosis records.")

    def _load_audio_models(self):
        """
        Load all necessary audio-related models:
        1. Load BEATs Tokenizer (for generating target token sequences)
        2. Load BEATs Feature Extractor (for extracting timbre embeddings)
        """
        try:
            # --- 1. Load BEATs Tokenizer (for generating target token sequences) ---
            beats_tokenizer_path = self.config['paths']['beats_tokenizer']
            if not os.path.exists(beats_tokenizer_path):
                raise FileNotFoundError(f"BEATs Tokenizer checkpoint not found at: {beats_tokenizer_path}")

            logger.info(f"Rank {self.cmd_args.local_rank}: Loading BEATs Tokenizer from {beats_tokenizer_path}...")
            tokenizer_checkpoint = torch.load(beats_tokenizer_path, map_location='cpu')
            tokenizer_cfg = TokenizersConfig(tokenizer_checkpoint['cfg'])
            self.beats_tokenizer = Tokenizers(tokenizer_cfg)
            self.beats_tokenizer.load_state_dict(tokenizer_checkpoint['model'])
            self.beats_tokenizer.eval()
            logger.info(
                f"Rank {self.cmd_args.local_rank}: BEATs Tokenizer loaded successfully and set to evaluation mode.")

            # --- 2. Load BEATs Feature Extractor (for timbre embedding) ---
            logger.info(f"Rank {self.cmd_args.local_rank}: Loading BEATs Feature Extractor...")
            beats_feature_extractor_path = self.config['paths']['beats_feature_extractor_checkpoint']
            if not os.path.exists(beats_feature_extractor_path):
                raise FileNotFoundError(
                    f"BEATs feature extractor checkpoint not found at: {beats_feature_extractor_path}")

            checkpoint = torch.load(beats_feature_extractor_path, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.beats_feature_extractor = BEATs(cfg)
            self.beats_feature_extractor.load_state_dict(checkpoint['model'])
            self.beats_feature_extractor.eval()
            logger.info(
                f"Rank {self.cmd_args.local_rank}: BEATs Feature Extractor loaded successfully from {beats_feature_extractor_path}.")

        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to load audio models: {e}\n{traceback.format_exc()}")
            raise

    def _setup_environment(self):
        rank = self.cmd_args.local_rank
        self.device = f'cuda:{rank}' if torch.cuda.is_available() and rank != -1 else 'cpu'
        try:
            self._load_metadata()
            self._load_audio_models()  # Call the new unified loading function
        except Exception as e:
            logger.error(f"Rank {rank}: Environment setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _setup_wandb(self):
        if self.config['logging']['wandb']['enabled']:
            try:
                wandb.init(
                    project=self.config['logging']['wandb']['project'],
                    name=self.config['logging']['wandb']['name'] + "_LLM",
                    config={**self.config, **vars(self.cmd_args)}
                )
                print("WandB initialized successfully")
            except Exception as e:
                logger.error(f"WandB initialization failed: {e}")
                self.config['logging']['wandb']['enabled'] = False

    def _setup_model_and_deepspeed(self):
        print(f"Loading LLM Tokenizer: {self.config['hyperparameters']['llm']['model_name']}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.config['hyperparameters']['llm']['model_name'],
            trust_remote_code=True
        )
        # First, extend the vocabulary (includes [PAD] / [AUDIO_i] / [BEATs_MASK] / [END] / [BEATs_*])
        self._extend_tokenizer_vocab()

        print(f"Loading LLM model: {self.config['hyperparameters']['llm']['model_name']}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config['hyperparameters']['llm']['model_name'],
            trust_remote_code=True
        )
        print("LLM model has been loaded in full precision (FP32).")

        # Need to resize after vocabulary extension
        model.resize_token_embeddings(len(self.llm_tokenizer))

        # ---- pad / eos fix ----
        if getattr(model.config, "pad_token_id", None) is None and self.llm_tokenizer.pad_token_id is not None:
            model.config.pad_token_id = self.llm_tokenizer.pad_token_id

        # Make [END] one of the generation terminators (coexists with the base model's original eos)
        end_id = self.llm_tokenizer.convert_tokens_to_ids('[END]')
        base_eos = getattr(model.config, "eos_token_id", None)
        if base_eos is None:
            model.config.eos_token_id = end_id
        else:
            if isinstance(base_eos, (list, tuple, set)):
                eos_ids = sorted(set(list(base_eos) + [end_id]))
            else:
                eos_ids = [base_eos] if base_eos != end_id else [base_eos]
            model.config.eos_token_id = eos_ids

        # Sync to generation_config (some models/library branches read this with priority)
        try:
            model.generation_config.eos_token_id = model.config.eos_token_id
        except Exception:
            pass

        # Register [END] as eos_token only if the tokenizer does not have one, to avoid overwriting the base model's </s>
        if getattr(self.llm_tokenizer, "eos_token_id", None) is None:
            try:
                self.llm_tokenizer.eos_token = '[END]'
            except Exception:
                pass

        # ---- Build "style prefix" projector and register it to the model ----
        llm_hidden_size = int(self.config['hyperparameters']['llm'].get('llm_hidden_size', 768))
        beats_feature_dim = int(self.config['hyperparameters']['llm'].get('beats_feature_size', 768))
        K = int(self.config['hyperparameters']['llm'].get('style_token_count', 16))

        # Register on the model, DeepSpeed will manage its parameters, saving/restoring will also be included
        model.style_token_count = K
        model.beats_feature_dim = beats_feature_dim

        # Pooling: Adaptive average pool, compresses BEATs sequence of any length into K segments
        model.style_pool = torch.nn.AdaptiveAvgPool1d(K)

        # Projector: Two-layer MLP + GELU, maps to LLM hidden dimension
        hidden = int(getattr(model.config, "hidden_size", llm_hidden_size))
        model.style_proj = torch.nn.Sequential(
            torch.nn.Linear(beats_feature_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden)
        )

        if self.config['hyperparameters']['llm'].get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        # Optimizer (includes LLM + style projector parameters)
        self.pytorch_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config['hyperparameters']['llm']['optimizer']['scheduler']['max_lr']),
            weight_decay=float(self.config['hyperparameters']['llm']['optimizer']['weight_decay']),
            betas=eval(str(self.config['hyperparameters']['llm']['optimizer']['betas']))
        )

        print(f"Rank {self.cmd_args.local_rank}: Initializing DeepSpeed engine...")
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=self.cmd_args,
            model=model,
            optimizer=self.pytorch_optimizer
        )
        self.device = self.model_engine.device
        print(f"DeepSpeed engine initialization complete. Model is on device: {self.device}")

        # Move external audio modules to the target device
        self.beats_tokenizer.to(self.device)
        self.beats_feature_extractor.to(self.device)

    def _extend_tokenizer_vocab(self):
        """
        Extend the vocabulary to include:
          - Task-specific Special Tokens: [DIAGNOSIS], [END], [BEATs_MASK], [PAD]
          - Style Prefix Tokens: [AUDIO_0]..[AUDIO_{K-1}]
          - BEATs Vocabulary Tokens: [BEATs_0]..[BEATs_{V-1}]
        And perform a pad fix (register tokenizer.pad_token). [END] is merged into eos_token_id in _setup_model_and_deepspeed.
        """
        print("Extending tokenizer vocabulary for the respiratory sound generation task...")

        K = int(self.config['hyperparameters']['llm'].get('style_token_count', 16))

        # Task-level base special tokens (removed old [AUDIO_EMBED])
        base_special_tokens = ['[DIAGNOSIS]', '[END]', '[BEATs_MASK]', '[PAD]']

        # List of style prefix tokens
        style_tokens = [f'[AUDIO_{i}]' for i in range(K)]

        # BEATs vocabulary size
        beats_vocab_size = self.config['hyperparameters']['flow']['vocab_size']
        beats_tokens = [f'[BEATs_{j}]' for j in range(beats_vocab_size)]

        # Add all at once
        self.llm_tokenizer.add_tokens(base_special_tokens + style_tokens + beats_tokens, special_tokens=True)

        # pad fix: ensure the tokenizer has a pad_token
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Record commonly used IDs
        self.special_token_ids = {
            'diagnosis': self.llm_tokenizer.convert_tokens_to_ids('[DIAGNOSIS]'),
            'end': self.llm_tokenizer.convert_tokens_to_ids('[END]'),
            'beats_mask': self.llm_tokenizer.convert_tokens_to_ids('[BEATs_MASK]'),
            'pad': self.llm_tokenizer.convert_tokens_to_ids('[PAD]'),
        }
        # List of IDs for the K style prefix tokens (in order)
        self.audio_style_token_ids = [self.llm_tokenizer.convert_tokens_to_ids(t) for t in style_tokens]

        print(f"Added {len(base_special_tokens) + len(style_tokens) + len(beats_tokens)} new tokens.")
        print(f"Vocabulary size after extension: {len(self.llm_tokenizer)}")

    def _setup_dataloaders(self, is_train=True):
        is_distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        if is_train:
            print("Initializing training data loader...")
            root_dir = self.config['data']['train_root']
            dataset = FusionDataset(root_dir=root_dir, config=self.config)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=True) if is_distributed else None
            self.train_loader = DataLoader(
                dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler, shuffle=(sampler is None),
                num_workers=self.config['data']['num_workers'], pin_memory=self.config['data']['pin_memory']
            )
            print(f"Training data loader initialized, contains {len(self.train_loader)} batches.")
        else:
            print("Initializing validation data loader...")
            root_dir = self.config['data']['val_root']
            dataset = FusionDataset(root_dir=root_dir, config=self.config)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False) if is_distributed else None
            self.val_loader = DataLoader(
                dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler, shuffle=False,
                num_workers=self.config['data']['num_workers'], pin_memory=self.config['data']['pin_memory']
            )
            print(f"Validation data loader initialized, contains {len(self.val_loader)} batches.")

    def setup_lr_scheduler(self):
        """Set up the OneCycleLR scheduler based on the configuration file."""
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Optimizer or training data loader not initialized, cannot create LR scheduler.")
            return
        print(f"Rank {self.cmd_args.local_rank}: Setting up LR scheduler...")
        try:
            scheduler_config = self.config['hyperparameters']['llm']['optimizer']['scheduler']
            num_epochs = self.config['hyperparameters']['llm']['num_epochs']
            gradient_accumulation_steps = self.model_engine.gradient_accumulation_steps()

            # Use a more accurate total steps calculation
            total_micro_batches = len(self.train_loader) * num_epochs
            total_steps = math.ceil(total_micro_batches / gradient_accumulation_steps)

            if total_steps <= 0:
                logger.error(
                    f"Rank {self.cmd_args.local_rank}: Calculated total steps ({total_steps}) is invalid, LR scheduler not created.")
                return

            print(f"Rank {self.cmd_args.local_rank}: Total steps for LR scheduler: {total_steps}")
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.pytorch_optimizer,
                max_lr=float(scheduler_config['max_lr']),
                total_steps=total_steps,
                pct_start=float(scheduler_config['pct_start']),
                div_factor=float(scheduler_config['div_factor']),
                final_div_factor=float(scheduler_config['final_div_factor'])
            )
            print(f"Rank {self.cmd_args.local_rank}: OneCycleLR scheduler created successfully.")
        except KeyError as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Failed to create LR scheduler. "
                         f"Missing key in config: {e}.")
            raise
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to set up LR scheduler: {e}\n{traceback.format_exc()}")
            raise

    def _prepare_batch_data(self, batch, is_train):
        """
        Prepares batch data for the "Disease + Style Prefix -> BEATs Token Sequence" task.

        Key changes:
          - Extract sequential features [B, T, D] using BEATs Feature Extractor;
          - Adaptive average pool to K segments, then through style_proj to get [B, K, H] style prefix embeddings;
          - Prompt is constructed as: "[DIAGNOSIS] <disease> [AUDIO_0]...[AUDIO_{K-1}]"
          - Target BEATs sequence still uses the [BEATs_*] sequence + [END] given by the tokenizer;
          - Randomly sample ~10% of positions t in the target interval, **and replace the input token at the previous position (t-1) with the [BEATs_MASK] embedding**,
            thus not feeding the ground truth of the masked segment to the model prematurely (aligns with causal LM's shift).
        """
        from pathlib import Path
        waveforms = batch['waveform'].to(self.device, non_blocking=True)
        file_paths = batch['file_path']
        batch_size = waveforms.size(0)

        with torch.no_grad():
            # --- 1) Sequential timbre features [B, T_feat, D_feat] ---
            audio_features, _ = self.beats_feature_extractor.extract_features(waveforms, padding_mask=None)
            # --- 2) Target BEATs Token sequence [B, L_tgt] ---
            flat_labels = self.beats_tokenizer.extract_labels(waveforms)
            target_beats_indices = flat_labels.view(batch_size, -1) if batch_size > 1 else flat_labels.unsqueeze(0)

        # --- 3) Style prefix: pooling + projection -> [B, K, H] ---
        # audio_features: [B, T, D] -> [B, D, T] to be compatible with 1D Pool
        feats_BDT = audio_features.transpose(1, 2)  # [B, D_feat, T_feat]
        pooled_BDK = self.model_engine.module.style_pool(feats_BDT)  # [B, D_feat, K]
        pooled_BKD = pooled_BDK.transpose(1, 2)  # [B, K, D_feat]
        style_embeds_BKH = self.model_engine.module.style_proj(pooled_BKD)  # [B, K, H]

        # --- 4) Construct text sequences ---
        K = int(self.model_engine.module.style_token_count)
        audio_prefix_text = " ".join([f"[AUDIO_{j}]" for j in range(K)])

        full_sequences_text = []
        prompt_text_only = []

        for i, file_path in enumerate(file_paths):
            filename = Path(file_path).name
            disease = self.filename_to_disease.get(filename, "Unknown Disease")

            prompt_part = f"[DIAGNOSIS] {disease} {audio_prefix_text}"
            target_part = "".join([f"[BEATs_{tok.item()}]" for tok in target_beats_indices[i]]) + "[END]"

            # Add a space in the middle to separate prompt and target, keeping it minimally invasive
            full_sequences_text.append(prompt_part + " " + target_part)
            prompt_text_only.append(prompt_part)

        # --- 5) Tokenize ---
        tokenized_full = self.llm_tokenizer(
            full_sequences_text, padding='longest', truncation=True,
            max_length=self.config['hyperparameters']['llm']['max_sequence_length'],
            return_tensors='pt'
        ).to(self.device)
        input_ids = tokenized_full['input_ids']  # [B, L]
        attention_mask = tokenized_full['attention_mask']  # [B, L]

        # Embeddings (will be partially replaced later)
        embed_layer = self.model_engine.module.get_input_embeddings()
        input_embeds = embed_layer(input_ids)  # [B, L, H]

        # --- 6) Replace the embeddings of the K [AUDIO_i] tokens with the style prefix embeddings [B, K, H] ---
        for b in range(batch_size):
            # Locate the position of each [AUDIO_i] (replace in order of i)
            for j, tok_id in enumerate(self.audio_style_token_ids):
                pos = (input_ids[b] == tok_id).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    continue
                idx = pos[0].item()
                if j < style_embeds_BKH.size(1):
                    input_embeds[b, idx, :] = style_embeds_BKH[b, j, :]

        # --- 7) Mask only the "preceding input" in the target interval to train the LM to predict the selected token ---
        # Find the prompt length (start of the target interval)
        tokenized_prompt = self.llm_tokenizer(prompt_text_only, padding='longest', return_tensors='pt').to(self.device)
        prompt_lengths = tokenized_prompt.attention_mask.sum(dim=1)  # [B]
        end_id = self.special_token_ids['end']
        mask_id = self.special_token_ids['beats_mask']

        # Get the embedding for [BEATs_MASK] to use for replacement
        beats_mask_embed = embed_layer.weight[mask_id].detach()

        mask_ratio = float(self.config['hyperparameters']['llm'].get('mask_ratio', 0.10))

        for b in range(batch_size):
            pl = int(prompt_lengths[b].item())
            # Target interval: [pl, end_pos), does not include [END]
            end_positions = (input_ids[b] == end_id).nonzero(as_tuple=True)[0]
            if end_positions.numel() == 0:
                continue
            end_pos = int(end_positions[0].item())

            # Predictable positions: t in [pl, end_pos-1]; we will replace the "preceding input" (t-1) for each selected t
            tgt_start = pl
            tgt_end = max(pl, end_pos - 1)
            tgt_len = max(0, tgt_end - tgt_start + 1)
            if tgt_len <= 1:
                continue

            # Sample about 10% of positions as "token indices t to be predicted"
            num_to_mask = max(1, int(round(tgt_len * mask_ratio)))
            # Exclude the first t=pl (because we replace t-1, and t=pl-1 belongs to the prompt, not recommended to change)
            candidate_indices = torch.arange(tgt_start + 1, end_pos, device=input_ids.device)
            if candidate_indices.numel() == 0:
                continue
            perm = torch.randperm(candidate_indices.numel(), device=input_ids.device)
            chosen = candidate_indices[perm[:num_to_mask]]

            # Replace the "preceding input" of each chosen token with the [BEATs_MASK] embedding
            prev_indices = (chosen - 1).tolist()
            for p in prev_indices:
                input_embeds[b, p, :] = beats_mask_embed

        # --- 8) Construct labels: mask out the prompt and padding ---
        labels = input_ids.clone()
        for b in range(labels.size(0)):
            pl = int(prompt_lengths[b].item())
            labels[b, :pl] = -100
        pad_id = self.llm_tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return input_embeds, attention_mask, labels

    def train(self):
        print("Starting LLM model training...")
        self._setup_dataloaders(is_train=True)
        self.setup_lr_scheduler()

        num_epochs = self.config['hyperparameters']['llm']['num_epochs']
        rank = self.cmd_args.local_rank

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()
            if self.train_loader.sampler is not None:
                self.train_loader.sampler.set_epoch(epoch)

            pbar = self.train_loader
            if rank <= 0:
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=150)

            for step, batch in enumerate(pbar):
                # *** Receive inputs_embeds ***
                inputs_embeds, attention_mask, labels = self._prepare_batch_data(batch, is_train=True)

                if (labels != -100).sum() == 0:
                    logger.warning(f"Rank {rank}: Skipping an invalid batch in training (all labels were masked).")
                    continue

                # *** Call model with inputs_embeds ***
                outputs = self.model_engine(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                self.model_engine.backward(loss)
                self.model_engine.step()

                if self.lr_scheduler and self.model_engine.is_gradient_accumulation_boundary():
                    self.lr_scheduler.step()

                self.global_step += 1

                if rank <= 0:
                    current_lr = self.pytorch_optimizer.param_groups[0]['lr']
                    perplexity = torch.exp(loss).item() if loss.isfinite() else float('inf')
                    pbar.set_postfix(
                        {'loss': f"{loss.item():.4f}", 'ppl': f"{perplexity:.2f}", 'lr': f"{current_lr:.2e}"})
                    if self.config['logging']['wandb']['enabled']:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/perplexity': perplexity,
                            'train/learning_rate': current_lr,
                            'global_step': self.global_step
                        })

            self.validate()

    def validate(self):
        rank = self.cmd_args.local_rank
        if rank <= 0:
            print("Starting validation...")

        if self.val_loader is None:
            self._setup_dataloaders(is_train=False)

        self.model_engine.eval()
        total_val_loss = 0.0
        total_batches = 0

        pbar = self.val_loader
        if rank <= 0:
            pbar = tqdm(self.val_loader, desc="Validating", ncols=120)

        with torch.no_grad():
            for batch in pbar:
                # *** Receive inputs_embeds ***
                inputs_embeds, attention_mask, labels = self._prepare_batch_data(batch, is_train=False)

                if (labels != -100).sum() == 0:
                    logger.warning(f"Rank {rank}: Skipping an invalid batch in validation.")
                    continue

                # *** Call model with inputs_embeds ***
                outputs = self.model_engine(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_val_loss += loss.item()
                total_batches += 1

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if world_size > 1:
            total_val_loss_tensor = torch.tensor(total_val_loss, device=self.device)
            total_batches_tensor = torch.tensor(total_batches, device=self.device)
            torch.distributed.all_reduce(total_val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_batches_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_val_loss = (total_val_loss_tensor / total_batches_tensor).item() if total_batches_tensor > 0 else float(
                'inf')
        else:
            avg_val_loss = total_val_loss / total_batches if total_batches > 0 else float('inf')

        # is_best_model is calculated based on the synchronized best_val_loss on all processes
        is_best_model = avg_val_loss < self.best_val_loss

        # rank 0 is responsible for printing and logging
        if rank <= 0:
            avg_perplexity = np.exp(avg_val_loss) if avg_val_loss != float('inf') else float('inf')
            print(f"Validation complete | Average Loss: {avg_val_loss:.4f} | Average Perplexity: {avg_perplexity:.2f}")

            if self.config['logging']['wandb']['enabled']:
                wandb.log({
                    'val/avg_loss': avg_val_loss,
                    'val/avg_perplexity': avg_perplexity,
                    'epoch': self.current_epoch + 1
                })

        # If it's the best model, all processes must execute the following block to stay synchronized
        if is_best_model:
            # Critical fix: all processes update self.best_val_loss to ensure they are in a consistent state for the next validation
            self.best_val_loss = avg_val_loss

            # Only rank 0 prints the save message
            if rank <= 0:
                print(f"New best model found, validation loss: {self.best_val_loss:.4f}. Saving...")

            # All processes call the save function. Since is_best_model is consistent across all processes,
            # save_checkpoint will be called by all processes simultaneously or skipped by all, thus avoiding deadlocks.
            # They now also have the same best_val_loss to generate a consistent filename.
            filename = f"best_model_loss_{self.best_val_loss:.4f}_epoch_{self.current_epoch + 1}.pth"
            self.save_checkpoint(filename)

    def save_checkpoint(self, filename):
        """
        Saves the best model checkpoint as a single .pth file, consistent with train_ast.py.
        - Only rank 0 performs the file saving operation.
        - The saved content includes model state, optimizer state, scheduler state, and client state.
        """
        # Ensure all processes are synchronized before starting to save
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.cmd_args.local_rank <= 0:
            try:
                checkpoint_file_path = self.checkpoint_dir / filename
                print(f"Preparing to save checkpoint to {checkpoint_file_path}")

                # Extract the unwrapped model state dictionary from the DeepSpeed engine
                model_state_dict = self.model_engine.module.state_dict()

                # Collect training state information
                client_state = {
                    'epoch': self.current_epoch + 1,
                    'global_step': self.global_step,
                    'best_val_loss': self.best_val_loss,
                    'config_yaml': self.config,
                }

                content_to_save = {
                    'model_state_dict': model_state_dict,
                    'pytorch_optimizer_state_dict': self.pytorch_optimizer.state_dict() if self.pytorch_optimizer else None,
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'client_state': client_state,
                }

                # Save as a single file
                torch.save(content_to_save, checkpoint_file_path)
                print(f"Best model checkpoint has been saved to: {checkpoint_file_path}")

            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                logger.error(traceback.format_exc())

        # Ensure all processes synchronize again after the save operation is complete
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Train LLM on AudioSet with DeepSpeed")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    setup_logger(args.local_rank)
    deepspeed.init_distributed()

    global trainer_global_object
    try:
        trainer_global_object = LLMTrainer(args.config, args)
        trainer_global_object.train()
    except Exception as e:
        logger.error("A critical error occurred during training!")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print("Training finished.")


if __name__ == "__main__":
    faulthandler.enable()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed = 666
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger = setup_logger()
    print("Determinism settings enabled (cudnn.benchmark=False, cudnn.deterministic=True).")

    main()
