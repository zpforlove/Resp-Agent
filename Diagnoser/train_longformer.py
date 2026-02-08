# -*- coding: utf-8 -*-
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb
from beats.BEATs import BEATs, BEATsConfig
from data_loader import FusionDataset
from utils import load_config

logger = None


def setup_logger(rank: int = -1):
    current_logger = logging.getLogger(__name__)
    if current_logger.hasHandlers():
        current_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    log_format = (f'%(asctime)s - RANK {rank} - %(name)s - %(levelname)s - %(message)s'
                  if rank != -1 else '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    current_logger.addHandler(handler)
    current_logger.setLevel(logging.INFO if rank in (-1, 0) else logging.WARNING)
    current_logger.propagate = False

    global logger
    logger = current_logger
    return logger


class LongformerTrainer:
    FIXED_AUDIO_STEPS = 496

    def __init__(self, config_path, cmd_args):
        self.cmd_args = cmd_args
        self.config = load_config(config_path)

        global logger
        logger = setup_logger(self.cmd_args.local_rank)

        # Components
        self.tokenizer = None
        self.model_engine = None
        self.optimizer = None
        self.pytorch_optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.device = None

        # Training State
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Audio
        self.beats_feature_extractor = None  # BEATs
        self.projection_layer = None  # BEATs -> hidden

        # Labels & Metadata
        self.filename_to_metadata = {}
        self.disease_tokens = []
        self.disease_to_index = {}
        self.index_to_disease = {}
        self.num_labels = 0

        # Text Placeholder Cache
        self.audio_embed_start_token_id = None
        self.special_token_ids = {}

        # Initialization
        self._setup_environment()
        if self.cmd_args.local_rank <= 0:
            self._setup_wandb()
        self._setup_model_and_deepspeed()

        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / 'longformer'
        if self.cmd_args.local_rank <= 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoint directory: {self.checkpoint_dir}")

    # -----------------------
    # Environment & Resources
    # -----------------------
    def _load_metadata(self):
        print("Loading audio_descriptions.jsonl metadata...")
        self.filename_to_metadata = {}
        unique_diseases = set()
        jsonl_path = "audio_descriptions.jsonl"
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        fn = rec.get("audio_filename")
                        disease = rec.get("disease")
                        desc = rec.get("description")
                        if fn and disease and desc:
                            self.filename_to_metadata[fn] = {"disease": disease, "description": desc}
                            unique_diseases.add(disease)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[ERROR] Failed to load metadata: {e}", file=sys.stderr)
            sys.exit(1)

        if not self.filename_to_metadata:
            print("[ERROR] Metadata map is empty, please check audio_descriptions.jsonl.", file=sys.stderr)
            sys.exit(1)

        self.disease_tokens = sorted(list(unique_diseases))
        self.disease_to_index = {d: i for i, d in enumerate(self.disease_tokens)}
        self.index_to_disease = {i: d for d, i in self.disease_to_index.items()}
        self.num_labels = len(self.disease_tokens)
        print(f"Metadata loaded: {len(self.filename_to_metadata)} records, {self.num_labels} classes.")

    def _load_audio_models(self):
        try:
            logger.info(f"RANK {self.cmd_args.local_rank}: Loading BEATs...")
            beats_ckpt = self.config['paths']['beats_feature_extractor_checkpoint']
            checkpoint = torch.load(beats_ckpt, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.beats_feature_extractor = BEATs(cfg)
            self.beats_feature_extractor.load_state_dict(checkpoint['model'])
            self.beats_feature_extractor.eval()
            logger.info("BEATs loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BEATs: {e}\n{traceback.format_exc()}")
            raise

    def _setup_environment(self):
        rank = self.cmd_args.local_rank
        self.device = f'cuda:{rank}' if torch.cuda.is_available() and rank != -1 else 'cpu'
        self._load_metadata()
        self._load_audio_models()

    def _setup_wandb(self):
        if self.config['logging']['wandb']['enabled']:
            try:
                wandb.init(
                    project=self.config['logging']['wandb']['project'],
                    name=self.config['logging']['wandb']['name'] + "_LONGFORMER",
                    config={**self.config, **vars(self.cmd_args)}
                )
            except Exception as e:
                logger.error(f"W&B initialization failed: {e}")
                self.config['logging']['wandb']['enabled'] = False

    # -----------------------
    # Model & DeepSpeed
    # -----------------------
    def _extend_tokenizer_vocab(self):
        # Register [DESCRIPTION] + 496 [AUDIO_EMBED_*] as non-splittable special tokens
        base_special_tokens = ['[DESCRIPTION]']
        audio_embed_tokens = [f'[AUDIO_EMBED_{i}]' for i in range(1, self.FIXED_AUDIO_STEPS + 1)]

        added = self.tokenizer.add_special_tokens({
            "additional_special_tokens": base_special_tokens + audio_embed_tokens
        })
        no_split = set(getattr(self.tokenizer, "unique_no_split_tokens", []))
        no_split.update(base_special_tokens + audio_embed_tokens)
        self.tokenizer.unique_no_split_tokens = list(no_split)

        self.special_token_ids = {
            'description': self.tokenizer.convert_tokens_to_ids('[DESCRIPTION]')
        }
        self.audio_embed_start_token_id = self.tokenizer.convert_tokens_to_ids('[AUDIO_EMBED_1]')
        self.audio_embed_ids = self.tokenizer.convert_tokens_to_ids(audio_embed_tokens)
        print(f"Added special tokens: {added}, current vocab size: {len(self.tokenizer)}")

    def _setup_model_and_deepspeed(self):
        lf_cfg = self.config['hyperparameters']['longformer']

        # 1) tokenizer
        print(f"Loading Longformer Tokenizer: {lf_cfg['model_name']} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(lf_cfg['model_name'], use_fast=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token:
                self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"

        # 2) Extend vocabulary
        self._extend_tokenizer_vocab()

        # 3) Classification model
        print(f"Loading Longformer classification model: {lf_cfg['model_name']} ...")
        model = AutoModelForSequenceClassification.from_pretrained(
            lf_cfg['model_name'],
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        model.resize_token_embeddings(len(self.tokenizer))

        # 4) Create projection layer: BEATs -> hidden (read dimension from embedding layer)
        hidden = model.get_input_embeddings().embedding_dim
        beats_feat_dim = int(lf_cfg['beats_feature_size'])
        self.projection_layer = torch.nn.Linear(beats_feat_dim, hidden)
        model.projection_layer = self.projection_layer  # Managed by DeepSpeed
        print(f"Audio projection layer: {beats_feat_dim} -> {hidden}")

        # 5) Optional gradient checkpointing
        if lf_cfg.get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        # 6) Optimizer
        self.pytorch_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lf_cfg['optimizer']['scheduler']['max_lr']),
            weight_decay=float(lf_cfg['optimizer']['weight_decay']),
            betas=eval(str(lf_cfg['optimizer']['betas']))
        )

        # 7) DeepSpeed
        print(f"RANK {self.cmd_args.local_rank}: Initializing DeepSpeed ...")
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=self.cmd_args,
            model=model,
            optimizer=self.pytorch_optimizer
        )
        self.device = self.model_engine.device
        self.beats_feature_extractor.to(self.device)
        print("DeepSpeed initialization complete.")

    # -----------------------
    # DataLoader & LR
    # -----------------------
    def _setup_dataloaders(self, is_train=True):
        is_distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        if is_train:
            root_dir = self.config['data']['train_root']
            dataset = FusionDataset(root_dir=root_dir, config=self.config)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=True) if is_distributed else None
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )
        else:
            root_dir = self.config['data']['val_root']
            dataset = FusionDataset(root_dir=root_dir, config=self.config)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False) if is_distributed else None
            self.val_loader = DataLoader(
                dataset,
                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler,
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )

    def setup_lr_scheduler(self):
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error("Optimizer or training data loader not initialized.")
            return
        cfg = self.config['hyperparameters']['longformer']['optimizer']['scheduler']
        num_epochs = self.config['hyperparameters']['longformer']['num_epochs']
        gas = self.model_engine.gradient_accumulation_steps()
        total_micro_batches = len(self.train_loader) * num_epochs
        total_steps = math.ceil(total_micro_batches / gas)
        if total_steps <= 0:
            logger.error("Total steps are invalid, cannot create OneCycleLR.")
            return
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.pytorch_optimizer,
            max_lr=float(cfg['max_lr']),
            total_steps=total_steps,
            pct_start=float(cfg['pct_start']),
            div_factor=float(cfg['div_factor']),
            final_div_factor=float(cfg['final_div_factor'])
        )

    # -----------------------
    # Align BEATs timesteps to 496
    # -----------------------
    @staticmethod
    def _align_to_fixed_steps(x, steps, mode="center"):
        B, T, D = x.shape
        if T == steps:
            return x
        if T > steps:
            if mode == "left":
                return x[:, :steps, :]
            elif mode == "right":
                return x[:, -steps:, :]
            else:
                start = (T - steps) // 2
                return x[:, start:start + steps, :]
        else:
            pad_len = steps - T
            pad = x.new_zeros(B, pad_len, D)
            return torch.cat([x, pad], dim=1)

    # -----------------------
    # Batch Preparation: Fixed 496 audio placeholders + Longformer global attention
    # -----------------------
    def _prepare_batch_data(self, batch, is_train):
        from pathlib import Path

        waveforms = batch['waveform'].to(self.device, non_blocking=True)
        file_paths = batch['file_path']

        # 1) BEATs -> projection -> 496 alignment
        with torch.no_grad():
            feats, _ = self.beats_feature_extractor.extract_features(waveforms, padding_mask=None)
        feats = self._align_to_fixed_steps(feats, self.FIXED_AUDIO_STEPS)
        projected = self.model_engine.module.projection_layer(feats)

        lf_cfg = self.config['hyperparameters']['longformer']
        max_seq_len = int(lf_cfg.get('max_sequence_length', 4096))
        max_text_tokens_cfg = int(lf_cfg.get('max_text_tokens', 128))
        text_drop_prob = float(lf_cfg.get('text_drop_prob', 0.3 if is_train else 0.0))
        audio_drop_prob = float(lf_cfg.get('audio_drop_prob', 0.1 if is_train else 0.0))
        audio_global_stride = int(lf_cfg.get('audio_global_stride', 8))  # audio anchor stride

        # special token ids
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        desc_tok_id = self.special_token_ids['description']
        audio_ids = self.audio_embed_ids  # length=496

        # Length budget: [CLS] [DESCRIPTION] <desc> <496×AUDIO> [SEP]
        overhead_tokens = 3
        desc_budget = max(0, max_seq_len - self.FIXED_AUDIO_STEPS - overhead_tokens)
        desc_budget = min(desc_budget, max_text_tokens_cfg)

        batch_ids, labels_vec, kept_idx = [], [], []
        for i, fp in enumerate(file_paths):
            fn = Path(fp).name
            meta = self.filename_to_metadata.get(fn)
            if not meta:
                continue
            raw_desc = str(meta["description"]).strip()
            disease = meta["disease"]
            if disease not in self.disease_to_index:
                continue

            # Truncate text only
            desc_ids = self.tokenizer.encode(raw_desc, add_special_tokens=False,
                                             truncation=True, max_length=desc_budget)
            seq_ids = [cls_id, desc_tok_id] + desc_ids + audio_ids + [sep_id]
            if len(seq_ids) > max_seq_len:
                over = len(seq_ids) - max_seq_len
                if over > 0 and len(desc_ids) > 0:
                    desc_ids = desc_ids[:max(0, len(desc_ids) - over)]
                    seq_ids = [cls_id, desc_tok_id] + desc_ids + audio_ids + [sep_id]
                if len(seq_ids) > max_seq_len:
                    seq_ids = seq_ids[:max_seq_len]

            batch_ids.append(seq_ids)
            labels_vec.append(self.disease_to_index[disease])
            kept_idx.append(i)

        if not batch_ids:
            return None, None, None, None

        projected = projected[kept_idx]

        # 3) Intra-batch padding
        max_len_in_batch = max(len(seq) for seq in batch_ids)
        input_ids_list, attn_masks_list = [], []
        for seq in batch_ids:
            pad_len = max_len_in_batch - len(seq)
            input_ids_list.append(seq + [pad_id] * pad_len)
            attn_masks_list.append([1] * len(seq) + [0] * pad_len)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)  # [B, L]
        attention_mask = torch.tensor(attn_masks_list, dtype=torch.long, device=self.device)  # [B, L]

        # Text drop (only between [DESCRIPTION] and [AUDIO_EMBED_1])
        work_input_ids = input_ids.clone()
        audio_start_id = self.audio_embed_start_token_id
        if is_train and text_drop_prob > 0.0:
            for i in range(work_input_ids.size(0)):
                ids = work_input_ids[i]
                pos_desc = (ids == desc_tok_id).nonzero(as_tuple=True)[0]
                pos_audio = (ids == audio_start_id).nonzero(as_tuple=True)[0]
                if len(pos_desc) == 0 or len(pos_audio) == 0:
                    continue
                start = int(pos_desc[0].item()) + 1
                end = int(pos_audio[0].item())
                if end > start:
                    span = torch.arange(start, end, device=self.device)
                    drop_mask = (torch.rand(span.numel(), device=self.device) < text_drop_prob)
                    if drop_mask.any():
                        ids[span[drop_mask]] = pad_id

        # Word embeddings
        input_embeds = self.model_engine.module.get_input_embeddings()(work_input_ids)  # [B, L, H]

        # Audio drop (randomly zero out timesteps)
        if is_train and audio_drop_prob > 0.0:
            Bk, T, H = projected.shape
            drop = (torch.rand(Bk, T, device=self.device) < audio_drop_prob).unsqueeze(-1)
            projected = projected.masked_fill(drop, 0.0)

        # Replace embeddings starting from [AUDIO_EMBED_1] with the 496 audio segments
        for i in range(work_input_ids.size(0)):
            ids = work_input_ids[i]
            pos_audio = (ids == audio_start_id).nonzero(as_tuple=True)[0]
            if len(pos_audio) == 0:
                continue
            start_idx = int(pos_audio[0].item())
            end_idx = min(start_idx + self.FIXED_AUDIO_STEPS, input_embeds.size(1))
            n = end_idx - start_idx
            if n > 0:
                input_embeds[i, start_idx:end_idx, :] = projected[i, :n, :]

        # Longformer's global attention:
        # Only make [CLS], [DESCRIPTION], and strided-sampled audio anchors global
        global_attention_mask = torch.zeros_like(attention_mask, dtype=torch.long)

        # First, set [CLS] and [DESCRIPTION]
        for i in range(work_input_ids.size(0)):
            ids = work_input_ids[i]
            pos_cls = (ids == cls_id).nonzero(as_tuple=True)[0]
            pos_desc = (ids == desc_tok_id).nonzero(as_tuple=True)[0]
            if len(pos_cls) > 0:
                global_attention_mask[i, int(pos_cls[0].item())] = 1
            if len(pos_desc) > 0:
                global_attention_mask[i, int(pos_desc[0].item())] = 1

        # Then, set strided-sampled audio anchors
        stride = max(1, int(audio_global_stride))
        Bk, L = attention_mask.shape
        for i in range(work_input_ids.size(0)):
            ids = work_input_ids[i]
            pos_audio = (ids == audio_start_id).nonzero(as_tuple=True)[0]
            if len(pos_audio) == 0:
                continue
            audio_start = int(pos_audio[0].item())
            audio_end = min(audio_start + self.FIXED_AUDIO_STEPS, L)
            T = max(0, audio_end - audio_start)
            if T <= 0:
                continue
            idx = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
            pick = (idx % stride == 0).long()  # (1, T)
            global_attention_mask[i, audio_start:audio_end] = pick.squeeze(0)

        labels = torch.tensor(labels_vec, dtype=torch.long, device=self.device)
        return input_embeds, attention_mask, global_attention_mask, labels

    # -----------------------
    # Training & Validation
    # -----------------------
    def train(self):
        print("Starting Longformer training (text + fixed 496 audio embeddings)...")
        self._setup_dataloaders(is_train=True)
        self.setup_lr_scheduler()

        num_epochs = self.config['hyperparameters']['longformer']['num_epochs']
        rank = self.cmd_args.local_rank

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()
            if self.train_loader.sampler is not None:
                self.train_loader.sampler.set_epoch(epoch)

            correct_running = 0
            total_running = 0

            it = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                      ncols=150) if rank <= 0 else self.train_loader
            for step, batch in enumerate(it):
                prepped = self._prepare_batch_data(batch, is_train=True)
                if prepped is None:
                    logger.warning("Skipping invalid batch (preprocessing failed)")
                    continue
                inputs_embeds, attention_mask, global_attention_mask, labels = prepped
                if labels is None or labels.numel() == 0:
                    logger.warning("Skipping invalid batch (labels are empty)")
                    continue

                outputs = self.model_engine(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                preds = outputs.logits.argmax(dim=-1)

                correct_running += (preds == labels).sum().item()
                total_running += labels.size(0)
                train_acc = correct_running / max(1, total_running)

                self.model_engine.backward(loss)
                self.model_engine.step()
                if self.lr_scheduler and self.model_engine.is_gradient_accumulation_boundary():
                    self.lr_scheduler.step()
                self.global_step += 1

                if rank <= 0:
                    lr = self.pytorch_optimizer.param_groups[0]['lr']
                    if isinstance(it, tqdm):
                        it.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{train_acc:.3f}", 'lr': f"{lr:.2e}"})
                    if self.config['logging']['wandb']['enabled']:
                        wandb.log({'train/loss': loss.item(), 'train/acc': train_acc, 'train/lr': lr,
                                   'global_step': self.global_step})

            self.validate()

    def validate(self):
        rank = self.cmd_args.local_rank
        if self.val_loader is None:
            self._setup_dataloaders(is_train=False)

        self.model_engine.eval()
        total_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0

        it = tqdm(self.val_loader, desc="Validating", ncols=120) if rank <= 0 else self.val_loader
        with torch.no_grad():
            for batch in it:
                prepped = self._prepare_batch_data(batch, is_train=False)
                if prepped is None:
                    continue
                inputs_embeds, attention_mask, global_attention_mask, labels = prepped
                if labels is None or labels.numel() == 0:
                    continue
                outputs = self.model_engine(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                preds = outputs.logits.argmax(dim=-1)

                total_loss += loss.item()
                total_batches += 1
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if ws > 1:
            def _reduce_sum(x):
                t = torch.tensor(x, device=self.device, dtype=torch.float32)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                return t.item()

            total_loss = _reduce_sum(total_loss)
            total_batches = _reduce_sum(total_batches)
            total_correct = _reduce_sum(total_correct)
            total_samples = _reduce_sum(total_samples)

        avg_val_loss = (total_loss / total_batches) if total_batches > 0 else float('inf')
        val_acc = (total_correct / total_samples) if total_samples > 0 else 0.0

        is_best = avg_val_loss < self.best_val_loss
        if rank <= 0:
            print(f"Validation | Average Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.3%}")
            if self.config['logging']['wandb']['enabled']:
                wandb.log({'val/loss': avg_val_loss, 'val/acc': val_acc, 'epoch': self.current_epoch + 1})
        if is_best:
            self.best_val_loss = avg_val_loss
            if rank <= 0:
                print(f"Found a better model (val_loss={self.best_val_loss:.4f}), saving...")
            filename = f"best_longformer_loss_{self.best_val_loss:.4f}_epoch_{self.current_epoch + 1}.pth"
            self._save_checkpoint(filename, val_acc)

    def _save_checkpoint(self, filename, val_acc):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.cmd_args.local_rank <= 0:
            try:
                path = self.checkpoint_dir / filename
                model_sd = self.model_engine.module.state_dict()
                client_state = {
                    'epoch': self.current_epoch + 1,
                    'global_step': self.global_step,
                    'best_val_loss': self.best_val_loss,
                    'val_accuracy': val_acc,
                    'num_labels': self.num_labels,
                    'label_mapping': self.index_to_disease,
                    'config_yaml': self.config,
                }
                torch.save({
                    'model_state_dict': model_sd,
                    'projection_layer_state_dict': self.model_engine.module.projection_layer.state_dict(),
                    'pytorch_optimizer_state_dict': self.pytorch_optimizer.state_dict() if self.pytorch_optimizer else None,
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'client_state': client_state,
                }, path)
                print(f"Saved: {path}")
            except Exception as e:
                logger.error(f"Save failed: {e}\n{traceback.format_exc()}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Longformer + Fixed 496 BEATs Embedding Classification Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    global logger
    logger = setup_logger(args.local_rank)
    deepspeed.init_distributed()

    try:
        trainer = LongformerTrainer(args.config, args)
        trainer.train()
    except Exception:
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

    print("Determinism settings are complete.")
    main()
