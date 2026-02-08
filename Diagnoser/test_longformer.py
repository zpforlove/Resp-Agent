# -*- coding: utf-8 -*-
import os
import re
import glob
import yaml
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from beats.BEATs import BEATs, BEATsConfig
import torch.nn as nn

# ============================
# Constants
# ============================
FIXED_AUDIO_STEPS = 496
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_TEXT_TOKENS = 128


def load_yaml_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_best_checkpoint(ckpt_dir: str) -> str:
    """
    Finds a file like best_longformer_loss_0.5479_epoch_5.pth under checkpoints/longformer and returns the one with the minimum loss.
    """
    pattern = os.path.join(ckpt_dir, 'best_longformer_loss_*.pth')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No weight files matching the pattern 'best_longformer_loss_*.pth' were found in {ckpt_dir}.")

    best_path, best_loss = None, float('inf')
    for f in files:
        m = re.search(r"best_longformer_loss_(\d+\.\d+)_epoch_\d+\.pth$", os.path.basename(f))
        if not m:
            continue
        loss = float(m.group(1))
        if loss < best_loss:
            best_loss, best_path = loss, f
    if best_path is None:
        raise RuntimeError("Could not parse loss value from filename, please check the saving naming convention.")
    print(f"Found best checkpoint: {os.path.basename(best_path)} (loss={best_loss:.4f})\nPath: {best_path}")
    return best_path


def load_jsonl_metadata(jsonl_path: str):
    meta = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fn = rec.get('audio_filename')
            disease = rec.get('disease')
            desc = rec.get('description')
            if fn and disease and desc:
                meta[fn] = {"disease": disease, "description": str(desc)}
    if not meta:
        raise RuntimeError(f"No valid entries (audio_filename+disease+description) were parsed from {jsonl_path}.")
    print(f"Metadata loading complete, {len(meta)} entries in total.")
    return meta


def safe_load_audio(path: str):
    try:
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            return None, 0
        waveform, sr = torchaudio.load(path)
        if waveform.numel() == 0:
            return None, 0
        # Mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sr
    except Exception:
        return None, 0


def resample_pad_or_trim(waveform: torch.Tensor, sr: int, target_sr: int, target_len_samples: int):
    if sr != target_sr:
        try:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        except Exception:
            return None
    cur = waveform.shape[1]
    if cur > target_len_samples:
        waveform = waveform[:, :target_len_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_len_samples - cur))
    return waveform


# ============================
# Test Set Dataset
# ============================
class LFAudioDataset(Dataset):
    def __init__(self, test_dir: str, meta_map: dict, disease_to_index: dict, target_sr: int, target_len_samples: int):
        self.test_dir = test_dir
        self.meta_map = meta_map
        self.d2i = disease_to_index
        self.target_sr = target_sr
        self.num_samples = target_len_samples
        self.wav_files = glob.glob(os.path.join(test_dir, '*.wav'))

        self.records = []  # list of (path, label_idx, description)
        skipped = 0
        for fp in self.wav_files:
            fn = os.path.basename(fp)
            meta = self.meta_map.get(fn)
            if not meta:
                skipped += 1
                continue
            disease = meta['disease']
            if disease not in self.d2i:
                skipped += 1
                continue
            self.records.append((fp, self.d2i[disease], meta['description']))
        if not self.records:
            raise RuntimeError("No files in the test set could be matched in the metadata (with disease+description).")
        print(f"Number of test samples: {len(self.records)}, skipped (no label/description): {skipped}")

        self.default_waveform = torch.zeros(self.num_samples)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fp, label_idx, desc = self.records[idx]
        wav, sr = safe_load_audio(fp)
        if wav is None:
            wav = self.default_waveform.clone().unsqueeze(0)
            return wav.squeeze(0), label_idx, fp, desc
        wav = resample_pad_or_trim(wav, sr, self.target_sr, self.num_samples)
        if wav is None:
            wav = self.default_waveform.clone().unsqueeze(0)
        return wav.squeeze(0), label_idx, fp, desc


# ============================
# Inference Wrapper: Longformer + 496 Audio Embeddings + BEATs
# ============================
class LongformerWithBEATsInfer:
    def __init__(self, config: dict, checkpoint_path: str, idx_to_disease: dict, device: torch.device):
        self.cfg = config
        self.device = device

        # 1) Read config from ckpt to ensure consistency
        pack = torch.load(checkpoint_path, map_location=device)
        client = pack.get('client_state', {})
        self.train_cfg = client.get('config_yaml', self.cfg)

        self.model_name = (
            self.train_cfg.get('hyperparameters', {})
            .get('longformer', {})
            .get('model_name', 'allenai/longformer-base-4096')
        )
        self.max_seq_len = (
            self.train_cfg.get('hyperparameters', {})
            .get('longformer', {})
            .get('max_sequence_length', DEFAULT_MAX_SEQ_LEN)
        )
        self.max_text_tokens = (
            self.train_cfg.get('hyperparameters', {})
            .get('longformer', {})
            .get('max_text_tokens', DEFAULT_MAX_TEXT_TOKENS)
        )
        self.beats_feat_dim = (
            self.train_cfg.get('hyperparameters', {})
            .get('longformer', {})
            .get('beats_feature_size', 768)
        )
        self.audio_global_stride = int(
            self.train_cfg.get('hyperparameters', {}).get('longformer', {}).get('audio_global_stride', 8)
        )  # ★ Added: stride for inference

        # 2) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token:
                self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = 'right'

        # Register [DESCRIPTION] + 496 [AUDIO_EMBED_*] tokens
        base_special_tokens = ['[DESCRIPTION]']
        audio_embed_tokens = [f'[AUDIO_EMBED_{i}]' for i in range(1, FIXED_AUDIO_STEPS + 1)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": base_special_tokens + audio_embed_tokens})
        no_split = set(getattr(self.tokenizer, 'unique_no_split_tokens', []))
        no_split.update(base_special_tokens + audio_embed_tokens)
        self.tokenizer.unique_no_split_tokens = list(no_split)

        self.desc_tok_id = self.tokenizer.convert_tokens_to_ids('[DESCRIPTION]')
        self.audio_embed_ids = self.tokenizer.convert_tokens_to_ids(audio_embed_tokens)
        self.audio_embed_start_id = self.tokenizer.convert_tokens_to_ids('[AUDIO_EMBED_1]')

        # 3) Model
        num_labels = int(client.get('num_labels', len(idx_to_disease)))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels, problem_type='single_label_classification'
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # ★ Read hidden dimension from the embedding layer
        hidden = self.model.get_input_embeddings().embedding_dim
        self.model.projection_layer = nn.Linear(self.beats_feat_dim, hidden)

        # Load weights (including projection_layer)
        missing, unexpected = self.model.load_state_dict(pack['model_state_dict'], strict=False)
        if 'projection_layer_state_dict' in pack:
            try:
                self.model.projection_layer.load_state_dict(pack['projection_layer_state_dict'], strict=False)
            except Exception as e:
                print(f"Failed to load projection_layer_state_dict: {e}")

        self.model.to(self.device).eval()

        # 4) BEATs
        beats_ckpt_path = self.train_cfg.get('paths', {}).get('beats_feature_extractor_checkpoint')
        if not beats_ckpt_path or not os.path.exists(beats_ckpt_path):
            raise FileNotFoundError(
                f"BEATs checkpoint not found: {beats_ckpt_path} (please specify in config.yaml under paths.beats_feature_extractor_checkpoint)"
            )
        ckpt = torch.load(beats_ckpt_path, map_location='cpu')
        bcfg = BEATsConfig(ckpt['cfg'])
        self.beats = BEATs(bcfg)
        self.beats.load_state_dict(ckpt['model'])
        self.beats.eval().to(self.device)

    @staticmethod
    def align_to_fixed_steps(x: torch.Tensor, steps: int = FIXED_AUDIO_STEPS, mode: str = 'center'):
        B, T, D = x.shape
        if T == steps:
            return x
        if T > steps:
            if mode == 'left':
                return x[:, :steps, :]
            if mode == 'right':
                return x[:, -steps:, :]
            start = (T - steps) // 2
            return x[:, start:start + steps, :]
        pad_len = steps - T
        pad = x.new_zeros(B, pad_len, D)
        return torch.cat([x, pad], dim=1)

    def prepare_batch(self, waveforms: torch.Tensor, descriptions: list):
        """
        Returns: (inputs_embeds, attention_mask, global_attention_mask)
        """
        B = waveforms.size(0)

        # 1) BEATs features and project -> [B, 496, H]
        with torch.no_grad():
            feats, _ = self.beats.extract_features(waveforms.to(self.device), padding_mask=None)
        feats = self.align_to_fixed_steps(feats, FIXED_AUDIO_STEPS)
        projected = self.model.projection_layer(feats)  # [B, 496, H]

        # 2) Text encoding & manual concatenation
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        overhead = 3
        desc_budget = max(0, self.max_seq_len - FIXED_AUDIO_STEPS - overhead)
        desc_budget = min(desc_budget, self.max_text_tokens)

        batch_ids = []
        for i in range(B):
            desc = descriptions[i] if isinstance(descriptions[i], str) else str(descriptions[i])
            desc_ids = self.tokenizer.encode(desc, add_special_tokens=False, truncation=True, max_length=desc_budget)
            seq = [cls_id, self.desc_tok_id] + desc_ids + self.audio_embed_ids + [sep_id]
            if len(seq) > self.max_seq_len:
                over = len(seq) - self.max_seq_len
                if over > 0 and len(desc_ids) > 0:
                    desc_ids = desc_ids[:max(0, len(desc_ids) - over)]
                    seq = [cls_id, self.desc_tok_id] + desc_ids + self.audio_embed_ids + [sep_id]
                if len(seq) > self.max_seq_len:
                    seq = seq[:self.max_seq_len]
            batch_ids.append(seq)

        max_len = max(len(s) for s in batch_ids)
        input_ids_list, attn_mask_list = [], []
        for seq in batch_ids:
            pad_len = max_len - len(seq)
            input_ids_list.append(seq + [pad_id] * pad_len)
        # attention mask
        attn_mask_list = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in batch_ids]

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        attn_mask = torch.tensor(attn_mask_list, dtype=torch.long, device=self.device)

        # 3) Word embeddings -> replace the 496 audio segments
        input_embeds = self.model.get_input_embeddings()(input_ids)  # [B, L, H]
        for i in range(input_embeds.size(0)):
            ids = input_ids[i]
            pos_audio = (ids == self.audio_embed_start_id).nonzero(as_tuple=True)[0]
            if len(pos_audio) == 0:
                continue
            s = int(pos_audio[0].item())
            e = min(s + FIXED_AUDIO_STEPS, input_embeds.size(1))
            n = e - s
            if n > 0:
                input_embeds[i, s:e, :] = projected[i, :n, :]

        # 4) Longformer global attention: CLS + DESCRIPTION + strided audio anchors
        global_attention_mask = torch.zeros_like(attn_mask, dtype=torch.long)

        # First, set [CLS] and [DESCRIPTION]
        for i in range(input_ids.size(0)):
            ids = input_ids[i]
            pos_cls = (ids == cls_id).nonzero(as_tuple=True)[0]
            pos_desc = (ids == self.desc_tok_id).nonzero(as_tuple=True)[0]
            if len(pos_cls) > 0:
                global_attention_mask[i, int(pos_cls[0].item())] = 1
            if len(pos_desc) > 0:
                global_attention_mask[i, int(pos_desc[0].item())] = 1

        # Sample audio anchors with stride
        stride = max(1, int(self.audio_global_stride))
        Bk, L = attn_mask.shape
        for i in range(input_ids.size(0)):
            ids = input_ids[i]
            pos_audio = (ids == self.audio_embed_start_id).nonzero(as_tuple=True)[0]
            if len(pos_audio) == 0:
                continue
            audio_start = int(pos_audio[0].item())
            audio_end = min(audio_start + FIXED_AUDIO_STEPS, L)
            T = max(0, audio_end - audio_start)
            if T <= 0:
                continue
            idx = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
            pick = (idx % stride == 0).long()  # (1, T)
            global_attention_mask[i, audio_start:audio_end] = pick.squeeze(0)

        return input_embeds, attn_mask, global_attention_mask

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                global_attention_mask: torch.Tensor, labels: torch.Tensor = None):
        with torch.no_grad():
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels
            )
        return out


# ============================
# Evaluation Logic
# ============================
def evaluate(model_wrap: LongformerWithBEATsInfer, dataloader: DataLoader, criterion, device, idx_to_disease: dict):
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc="Testing on Test Set")
    for waveforms, labels, file_paths, descs in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        inputs_embeds, attn_mask, gmask = model_wrap.prepare_batch(waveforms, descs)
        out = model_wrap.forward(inputs_embeds, attn_mask, gmask, labels)

        loss = out.loss
        logits = out.logits
        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item())
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    present_labels = sorted(list(set(all_labels) | set(all_preds)))
    target_names = [idx_to_disease[i] for i in present_labels]
    report = classification_report(all_labels, all_preds, labels=present_labels, target_names=target_names,
                                   zero_division=0)
    return avg_loss, acc, f1, report


# ============================
# Main Entry Point
# ============================
def main():
    # Basic paths
    CONFIG_PATH = 'config.yaml'
    TEST_JSONL = 'audio_descriptions.jsonl'
    TEST_WAV_DIR = '/mnt/data/RespAgent/Fusion/dataset/test'
    CKPT_DIR = 'checkpoints/longformer'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = load_yaml_config(CONFIG_PATH)
    target_sr = cfg.get('audio', {}).get('sample_rate', 16000)
    target_len_samples = cfg.get('audio', {}).get('max_length', target_sr * 10)

    best_ckpt = find_best_checkpoint(CKPT_DIR)
    pack = torch.load(best_ckpt, map_location='cpu')
    client = pack.get('client_state', {})

    idx_to_disease = client.get('label_mapping')
    if not idx_to_disease:
        raise RuntimeError("Missing label_mapping (idx->disease) in the checkpoint's client_state.")
    disease_to_index = {v: int(k) for k, v in idx_to_disease.items()}

    meta = load_jsonl_metadata(TEST_JSONL)
    dataset = LFAudioDataset(
        test_dir=TEST_WAV_DIR,
        meta_map=meta,
        disease_to_index=disease_to_index,
        target_sr=target_sr,
        target_len_samples=target_len_samples
    )

    batch_size = int(cfg.get('hyperparameters', {}).get('longformer', {}).get('eval_batch_size', 4))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    infer = LongformerWithBEATsInfer(cfg, best_ckpt, idx_to_disease, device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, report = evaluate(infer, loader, criterion, device, idx_to_disease)

    print("\n--- Test Set Evaluation Results ---")
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Macro: {test_f1:.4f}")

    print("\n--- Classification Report ---")
    print(report)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
