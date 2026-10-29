# -*- coding: utf-8 -*-
"""RICE | ALL SEVEN TRAITS | FIXED C=256 | adaptive_pos only

Expected files (script beside data/):
  data/maf_0.05_ld_filtered_genotypes.txt
  data/rice299_LDG.csv
  data/rice299_FL.csv
  data/rice299_MAT.csv
  data/rice299_PH.csv
  data/rice299_TILLER.csv
  data/rice299_YLD.csv
  data/rice299_Kg_ha.csv

The genotype file is retained from the uploaded Rice LD-filtered script.
Phenotypes are joined to genotype samples by sample ID, not row position.
The filename prefix "rice299" is retained as supplied; the actual aligned
sample count is measured and recorded from the files at runtime.

ASCM: adaptive chunk weighting + learnable positional encoding (adaptive_pos).
Chunk size is FIXED at C=256. 10-fold KFold(random_state=42), train-only
imputation/scaling, 150 epochs per fold, no early stopping or validation split.
Exports out-of-fold predictions and supplementary prediction metrics.

Example:
  python PerfVAE_Rice_AllTraits_C256_adaptive_pos_metrics.py --check-data-only
  python PerfVAE_Rice_AllTraits_C256_adaptive_pos_metrics.py
"""
from __future__ import annotations

import math
import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import spearmanr, t as student_t, wilcoxon
from tqdm import tqdm

try:
    from performer_pytorch import Performer
except ImportError:
    Performer = None  # Raise a clear error in main() when missing.

# Use the location of the .py file (NOT os.getcwd()).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = SCRIPT_DIR / "data"

# ============================================================================
# CENTRAL CONFIGURATION -- all parameters are changed HERE, not on the CLI.
# ============================================================================
CONFIG = {
    # New LD-filtered Rice genotype (actual sample count is checked at runtime).
    # Two phenotypes from the early Rice data processing convention.
    "traits": ["LDG", "FL", "MAT", "PH", "TILLER", "YLD", "Kg_ha"],
    "data_dir": str(DATA_DIR_DEFAULT),

    # Run ONLY the combined ASCM configuration (adaptive weights + learnable
    # positional encoding), across the requested chunk-size sensitivity grid.
    "design": "full",
    "chunk_sizes": [256],
    "ablation_c": 256,
    "variants": ["adaptive_pos"],
    "weight_hidden": 64,
    "weight_temperature": 1.0,
    "export_chunk_weights": False,
    "bootstrap_replicates": 1000,
    # Rice settings aligned with the current EAAI manuscript protocol.
    "loss_weights": {"pred": 1.5, "recon": 1.0, "kl": 0.8},

    # FIXED experiment protocol: 10 folds x 150 epochs. No early stop,
    # no inner validation split. The FINAL epoch model is evaluated.
    "folds": 10,
    "max_epochs": 150,
    "batch_size": 48,
    "seed": 42,
    "dropout": 0.4,
    "lr": 1e-4,
    "max_lr": 1e-4,
    "weight_decay": 0.0,
    "latent_dim": 128,
    "performer_dim": 256,
    "device": "cuda",     # CUDA_VISIBLE_DEVICES=0 makes this physical GPU 0.

    # Results stored separately from previous early-stopping experiments.
    "out_dir": str(SCRIPT_DIR / "ASCM_results_Rice_AllTraits_C256_adaptive_pos_FIXED150"),
    "resume": True,              # Skip COMPLETED folds only when config matches.
    "keep_checkpoints": False,   # True saves the FINAL epoch model per fold.
    "no_progress": True,        # Keep nohup logs readable.
    "epoch_log_interval": 0,    # 0 = log ONE result per fold only (no per-epoch log).
    "no_plot": False,
    "allow_phenotype_row_order": False,  # Intentionally disabled: strict sample-ID join only.
    "minus_one_mode": "missing",      # OLD PerfVAE.py: X_raw[X_raw == -1] = np.nan
    "phenotype_column_overrides": {"LDG": None, "FL": None, "MAT": None, "PH": None, "TILLER": None, "YLD": None, "Kg_ha": None},
    "check_data_only": False,      # True: check new Rice data without training.
    "verbose_data": False,        # True: print file headers and ID alignment diagnostics.
}

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("._") or "unnamed"


def flush_gpu(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GeneticAwareLoss(nn.Module):
    """Loss weights are explicit in CONFIG; defaults preserve the uploaded script."""
    def __init__(self, weights=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.weights = dict(weights or {"pred": 1.5, "recon": 1.0, "kl": 0.01})

    def forward(self, x_recon, x, y_pred, y, mu, logvar, aux_logits=None, aux_targets=None):
        pred = self.mse(y_pred, y)
        recon = self.mse(x_recon, x)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        aux = torch.tensor(0.0, device=x.device)
        if aux_logits is not None and aux_targets is not None:
            aux = self.ce(aux_logits, aux_targets.long())
        return (self.weights["pred"] * pred
                + self.weights["recon"] * recon
                + self.weights["kl"] * kl
                + 0.0 * aux)


class AdaptiveChunkedSNP_Performer(nn.Module):
    """Fixed SNP chunks with OPTIONAL sample-dependent weights and positions.

    Learned chunk importance is NOT a learned chunk boundary/length and is NOT
    itself evidence of causal SNP function.
    """
    def __init__(self, snp_dim, chunk_size=256, performer_dim=256, dropout=0.4,
                 adaptive_weighting=False, learnable_position=False,
                 weight_hidden=64, weight_temperature=1.0):
        super().__init__()
        self.snp_dim = int(snp_dim)
        self.chunk_size = int(chunk_size)
        self.pad_n = (-snp_dim) % self.chunk_size
        self.num_chunks = (snp_dim + self.pad_n) // self.chunk_size
        self.adaptive_weighting = bool(adaptive_weighting)
        self.learnable_position = bool(learnable_position)
        self.weight_temperature = float(weight_temperature)
        if self.weight_temperature <= 0:
            raise ValueError("weight_temperature must be positive")
        self.padding_layer = nn.ZeroPad1d((0, self.pad_n)) if self.pad_n else None
        self.chunk_norm = nn.LayerNorm(self.chunk_size)

        if self.adaptive_weighting:
            # Shared scoring network evaluated independently for each chunk
            # and each genotype. Last linear layer starts at zero, yielding
            # uniform weights (w_i=1) for the first forward pass.
            # fork_rng prevents these extra parameters from consuming the
            # random sequence used to initialize the submitted backbone.
            # Hence matching fold seeds give identical shared initial weights
            # in original and enhanced variants, enabling a fair ablation.
            with torch.random.fork_rng(devices=[]):
                self.chunk_scorer = nn.Sequential(
                    nn.Linear(self.chunk_size, int(weight_hidden)), nn.GELU(),
                    nn.Linear(int(weight_hidden), 1)
                )
                nn.init.zeros_(self.chunk_scorer[-1].weight)
                nn.init.zeros_(self.chunk_scorer[-1].bias)

        if self.learnable_position:
            # Learned absolute CHUNK-index embeddings; shape [1,N,C].
            # Zero init preserves the submitted pathway at step zero.
            self.chunk_position = nn.Parameter(
                torch.zeros(1, self.num_chunks, self.chunk_size)
            )

        self.performer = Performer(
            dim=self.chunk_size, depth=2, heads=4, dim_head=32,
            ff_dropout=dropout, attn_dropout=dropout, causal=False
        )
        self.output_projection = nn.Linear(2 * self.chunk_size, performer_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def chunk_and_normalize(self, x):
        if self.padding_layer is not None:
            x = self.padding_layer(x.unsqueeze(1)).squeeze(1)
        return self.chunk_norm(x.reshape(x.size(0), self.num_chunks, self.chunk_size))

    def get_chunk_weights(self, x):
        """Returns per-sample weights [B,N,1]; None in non-adaptive models.

        w_i = N * softmax(score_i / temperature), so mean_i(w_i)=1.
        The final zero-padded SNP positions are excluded from the *scorer*
        features, but unchanged original normalized input goes to Performer.
        """
        if not self.adaptive_weighting:
            return None
        h = self.chunk_and_normalize(x)
        if self.pad_n:
            h = h.clone()
            h[:, -1, -self.pad_n:] = 0.0
        raw_scores = self.chunk_scorer(h)
        # FP32 scores for numerical stability, even under mixed precision.
        weights = torch.softmax(raw_scores.float() / self.weight_temperature, dim=1)
        weights = weights * self.num_chunks
        return weights.to(dtype=h.dtype)

    def forward(self, x):
        h = self.chunk_and_normalize(x)
        if self.adaptive_weighting:
            # Score from normalized chunks, not from the final feature vector.
            # Do not invoke get_chunk_weights: that would normalize twice.
            scorer_input = h
            if self.pad_n:
                scorer_input = h.clone()
                scorer_input[:, -1, -self.pad_n:] = 0.0
            scores = self.chunk_scorer(scorer_input)
            weights = torch.softmax(scores.float() / self.weight_temperature, dim=1)
            h = h * (self.num_chunks * weights).to(h.dtype)
        if self.learnable_position:
            h = h + self.chunk_position
        h = self.performer(h)
        h = h.transpose(1, 2)
        pooled = torch.cat(
            [self.avg_pool(h).squeeze(-1), self.max_pool(h).squeeze(-1)], dim=1
        )
        return self.output_projection(pooled)


VARIANTS = {

    "adaptive_pos": (True, True),
}


def make_config_grid(args):
    chunks = sorted(args.chunk_sizes)
    c0 = args.ablation_c
    if args.design in ("efficient", "ablation") and c0 not in chunks:
        raise ValueError(f"Ablation chunk size {c0} must be listed in CONFIG[chunk_sizes]")
    if args.design == "efficient":
        configs = [(mode, c) for mode in ("original", "adaptive_pos") for c in chunks]
        configs += [("adaptive", c0), ("positional", c0)]
    elif args.design == "sensitivity":
        configs = [(mode, c) for mode in ("original", "adaptive_pos") for c in chunks]
    elif args.design == "ablation":
        configs = [(mode, c0) for mode in VARIANTS]
    else:
        configs = [(mode, c) for mode in VARIANTS for c in chunks]
    if args.variants:
        configs = [(mode, c) for mode, c in configs if mode in args.variants]
    if not configs:
        raise ValueError("No configurations selected; check CONFIG[design] and CONFIG[variants]")
    return configs


class AdaptiveChunkedSNP_Performer_Decoder(nn.Module):
    def __init__(self, latent_dim, snp_dim, chunk_size=256, performer_dim=256, dropout=0.4):
        super().__init__()
        self.snp_dim = snp_dim
        self.chunk_size = int(chunk_size)
        self.pad_n = (-snp_dim) % self.chunk_size
        self.num_chunks = (snp_dim + self.pad_n) // self.chunk_size
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, performer_dim),
            nn.BatchNorm1d(performer_dim), nn.GELU()
        )
        self.to_seq = nn.Sequential(
            nn.Linear(performer_dim, self.num_chunks * self.chunk_size),
            nn.BatchNorm1d(self.num_chunks * self.chunk_size), nn.GELU()
        )
        self.performer = Performer(
            dim=self.chunk_size, depth=2, heads=4, dim_head=32,
            ff_dropout=dropout, attn_dropout=dropout, causal=False
        )
        self.final_head = nn.Linear(self.chunk_size, self.chunk_size)
        # Kept to match the submitted model, even though its loss weight is zero.
        self.aux_classifier = nn.Linear(self.chunk_size, 3 * self.chunk_size)

    def forward(self, z, sample_indices=None):
        b = z.size(0)
        h = self.latent_proj(z)
        h = self.to_seq(h).reshape(b, self.num_chunks, self.chunk_size)
        h = self.performer(h)
        x_recon = self.final_head(h).reshape(b, -1)[:, :self.snp_dim]
        aux_logits = None
        if sample_indices is not None:
            aux_logits = self.aux_classifier(h).reshape(b, -1, 3)
            aux_logits = aux_logits[:, sample_indices, :].reshape(-1, 3)
        return x_recon, aux_logits


class EnhancedGeneticModel(nn.Module):
    def __init__(self, snp_dim, chunk_size=256, latent_dim=128, performer_dim=256, dropout=0.4,
                 adaptive_weighting=False, learnable_position=False,
                 weight_hidden=64, weight_temperature=1.0):
        super().__init__()
        self.snp_performer = AdaptiveChunkedSNP_Performer(
            snp_dim, chunk_size=chunk_size, performer_dim=performer_dim, dropout=dropout,
            adaptive_weighting=adaptive_weighting,
            learnable_position=learnable_position,
            weight_hidden=weight_hidden, weight_temperature=weight_temperature
        )
        # EXACT layer widths and GELU placements from perfVAE0530.py.
        self.linear_shortcut = nn.Sequential(
            nn.Linear(snp_dim, 4096), nn.BatchNorm1d(4096), nn.GELU(),
            nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 64)
        )
        self.fc_mu = nn.Linear(performer_dim + 64, latent_dim)
        self.fc_logvar = nn.Linear(performer_dim + 64, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.decoder = AdaptiveChunkedSNP_Performer_Decoder(
            latent_dim, snp_dim, chunk_size=chunk_size,
            performer_dim=performer_dim, dropout=dropout
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x, sample_indices=None):
        h_performer = self.snp_performer(x)
        h_shortcut = self.linear_shortcut(x)
        h = torch.cat([h_performer, h_shortcut], dim=1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        y_pred = self.predictor(z)
        x_recon, aux_logits = self.decoder(z, sample_indices)
        return x_recon, y_pred, mu, logvar, aux_logits


# ========================= New Rice genotype/phenotype loader ====================
# Sample ID must be present as the FIRST column (early PerfVAE.py assumption).
# Do not use positional extraction on the joined dataframe: it can omit SNP columns.
RICE_TRAITS = {
    "LDG": "rice299_LDG.csv",
    "FL": "rice299_FL.csv",
    "MAT": "rice299_MAT.csv",
    "PH": "rice299_PH.csv",
    "TILLER": "rice299_TILLER.csv",
    "YLD": "rice299_YLD.csv",
    "Kg_ha": "rice299_Kg_ha.csv",
}
RICE_GENO_FILE = "maf_0.05_ld_filtered_genotypes.txt"
RICE_ID_ALIASES = {"entry", "id", "iid", "sample", "sampleid", "accession", "line",
                   "taxa", "taxon", "individual", "individualid", "genotypeid"}


def locate_rice_file(data_dir, name):
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {root}. Put the script next to data/ or change CONFIG['data_dir']")
    hits = [p for p in root.rglob('*') if p.is_file() and p.name.casefold() == name.casefold()]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly ONE file {name!r} under {root}; found: {[str(p) for p in hits]}")
    return hits[0]


def discover_experiments(data_dir, traits):
    geno = locate_rice_file(data_dir, RICE_GENO_FILE)
    exps = []
    for trait in traits:
        if trait not in RICE_TRAITS:
            raise ValueError(f"Unexpected trait {trait!r}. Supported traits: {list(RICE_TRAITS)}")
        pheno = locate_rice_file(data_dir, RICE_TRAITS[trait])
        exps.append(dict(dataset="Rice", trait=trait, geno=str(geno),
                         pheno=str(pheno), kind="rice"))
    return exps


def _column_clean(c):
    return str(c).replace("\ufeff", "").strip().strip('"').strip("'").strip()


def _alias_clean(c):
    return re.sub(r"[^a-z0-9]+", "", _column_clean(c).casefold())


def _detect_sep(path):
    with Path(path).open('r', encoding='utf-8-sig', errors='replace') as fh:
        line = fh.readline().rstrip('\r\n')
    if not line:
        raise ValueError(f"Empty header in {path}")
    counts = {'\t': line.count('\t'), ',': line.count(','), ';': line.count(';')}
    delimiter = max(counts, key=counts.get)
    return delimiter if counts[delimiter] > 0 else r'\s+'


def _read_rice_table(path, verbose=False):
    sep = _detect_sep(path)
    # Read the header first, then force the first column (sample IDs) to strings.
    # This preserves IDs such as "001" instead of allowing pandas to coerce them to 1.
    header = pd.read_csv(path, sep=sep, engine='c', encoding='utf-8-sig',
                         nrows=0, low_memory=False)
    if len(header.columns) < 2:
        raise ValueError(f"{path}: only {len(header.columns)} column(s); check delimiter and header")
    first_col = header.columns[0]
    df = pd.read_csv(path, sep=sep, engine='c', encoding='utf-8-sig',
                     low_memory=False, dtype={first_col: str})
    df.columns = [_column_clean(c) for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError(f"Duplicate column names in {path}: {df.columns[:10].tolist()}")
    if df.shape[1] < 2:
        raise ValueError(f"{path}: only {df.shape[1]} column(s); check delimiter and header")
    if verbose:
        print(f"[READ] {Path(path).name} shape={df.shape}, first columns={df.columns[:6].tolist()}", flush=True)
    return df


def _id_col(df, path):
    # Early PerfVAE.py uses first column as ID; retain it but validate afterwards.
    chosen = df.columns[0]
    candidate = [c for c in df.columns if _alias_clean(c) in RICE_ID_ALIASES]
    if chosen not in candidate and candidate:
        raise ValueError(f"{path}: first column {chosen!r} is not an ID alias, but found ID column(s) {candidate}. "
                         "The early script assumes the FIRST column contains IDs; inspect the file before continuing.")
    print(f"[ID] {Path(path).name}: using FIRST column {chosen!r}", flush=True)
    return chosen


def _id_key(value):
    if pd.isna(value):
        return ''
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    text = str(value).replace('\ufeff', '').strip()
    # Handle Excel-generated integer IDs displayed with a .0 suffix; preserve others.
    if re.fullmatch(r'[0-9]+\.0+', text):
        text = text.split('.')[0]
    return text


def _normalize_id(df, col, source):
    vals = df[col].map(_id_key)
    if (vals == '').any() or vals.duplicated().any():
        examples = vals[vals.duplicated(keep=False)].head(6).tolist()
        raise ValueError(f"{source}: empty/duplicated sample IDs: {examples}")
    out = df.copy()
    out['__RiceSampleID__'] = vals
    return out


def load_dataset(exp, allow_phenotype_row_order=False, verbose=False, minus_one_mode='missing',
                 phenotype_column_overrides=None):
    """New genotype source; join by IDs only, never by row order.

    -1 is treated as missing by DEFAULT to match user's early PerfVAE.py.
    For genotype coding where -1 is a genuine allele, set minus_one_mode='valid'
    AFTER inspecting the actual input values. This changes the protocol.
    Imputation will be fitted ONLY on outer-fold training data.
    """
    if allow_phenotype_row_order:
        raise ValueError('Row-order phenotype alignment has been intentionally disabled for this redo.')
    if minus_one_mode not in ('missing', 'valid'):
        raise ValueError('minus_one_mode must be missing or valid')
    geno, pheno, trait = Path(exp['geno']), Path(exp['pheno']), exp['trait']
    gd = _read_rice_table(geno, verbose=verbose)
    pd_ = _read_rice_table(pheno, verbose=verbose)
    geno_id = _id_col(gd, geno.name)
    pheno_id = _id_col(pd_, pheno.name)
    snp_cols = [c for c in gd.columns if c != geno_id]
    if len(snp_cols) < 2:
        raise ValueError(f'{geno.name}: expected many SNP columns, found {len(snp_cols)}')

    override = (phenotype_column_overrides or {}).get(trait)
    if override:
        matches = [c for c in pd_.columns if c.casefold() == str(override).casefold()]
        if len(matches) != 1:
            raise ValueError(f'{pheno.name}: override {override!r} not present. Headers: {pd_.columns[:20].tolist()}')
        trait_col = matches[0]
    else:
        matches = [c for c in pd_.columns if _alias_clean(c) == trait.casefold()]
        if len(matches) == 1 and matches[0] != pheno_id:
            trait_col = matches[0]
        elif len(pd_.columns) == 2:
            trait_col = pd_.columns[1]
            print(f'[INFO] {pheno.name}: using second column {trait_col!r} as {trait}, as in old PerfVAE.py', flush=True)
        else:
            raise ValueError(f'{pheno.name}: ambiguous trait column for {trait}. '
                             f'Headers: {pd_.columns[:18].tolist()}. '
                             f'Specify CONFIG["phenotype_column_overrides"]["{trait}"]')

    gd = _normalize_id(gd, geno_id, geno.name)
    phenotype = _normalize_id(pd_[[pheno_id, trait_col]].copy(), pheno_id, pheno.name)
    phenotype = phenotype[['__RiceSampleID__', trait_col]].rename(columns={trait_col:'__TraitValue__'})
    genotype_ids = set(gd['__RiceSampleID__'])
    phenotype_ids = set(phenotype['__RiceSampleID__'])
    common = genotype_ids & phenotype_ids
    only_genotype = len(genotype_ids - phenotype_ids)
    only_phenotype = len(phenotype_ids - genotype_ids)
    print(f'[ALIGN] trait={trait}, genotype rows={len(gd)}, phenotype rows={len(phenotype)}, '
          f'shared IDs={len(common)}, genotype-only={only_genotype}, phenotype-only={only_phenotype}', flush=True)
    if len(common) < 30:
        raise ValueError(f'{trait}: only {len(common)} matched IDs; first genotype IDs={gd[geno_id].head(8).tolist()}, '
                         f'first phenotype IDs={pd_[pheno_id].head(8).tolist()}')
    if only_genotype or only_phenotype:
        print(f'[NOTE] {trait}: unmatched IDs are excluded by inner join; check counts above.', flush=True)
    # Preserve genotype row order so traits with the same matched samples get the same CV ordering.
    merged = gd[['__RiceSampleID__'] + snp_cols].merge(phenotype, on='__RiceSampleID__',
                                                       how='inner', sort=False, validate='one_to_one')
    merged['__TraitValue__'] = pd.to_numeric(merged['__TraitValue__'], errors='coerce')
    before = len(merged)
    merged = merged.loc[merged['__TraitValue__'].notna()].copy()
    if len(merged) < 30:
        raise ValueError(f'{trait}: fewer than 30 aligned nonmissing phenotypes')
    X = merged[snp_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    y = merged['__TraitValue__'].to_numpy(dtype=np.float32).reshape(-1,1)
    minus_one_count = int(np.count_nonzero(X == -1))
    if minus_one_mode == 'missing':
        X[X == -1] = np.nan
    if np.isinf(X).any() or not np.isfinite(y).all():
        raise ValueError('Infinite genotype or phenotype values found; inspect data source')
    n_missing = int(np.isnan(X).sum())
    n_empty = int(np.isnan(X).all(axis=0).sum())
    if n_empty:
        raise ValueError(f'{n_empty} SNP columns are missing in ALL aligned samples; '
                         'input SNP matrix must be checked rather than silently dropping markers.')
    if not np.isfinite(y.std()) or y.std() < 1e-10:
        raise ValueError(f'{trait}: trait has zero/invalid variance')
    meta = {
        'Data_Kind': 'rice_ld_filtered_new_source', 'Genotype_File': geno.name,
        'Phenotype_File': pheno.name, 'Genotype_ID_Column': geno_id,
        'Phenotype_ID_Column': pheno_id, 'Phenotype_Value_Column': trait_col,
        'Genotype_Samples': len(gd), 'Phenotype_Samples': len(pd_),
        'Aligned_Samples': len(X), 'Dropped_Missing_Phenotypes': before-len(merged),
        'SNP_Dimension_D': X.shape[1], 'Missing_Cells_After_Coding': n_missing,
        'Minus_One_Cells_In_Raw_Input': minus_one_count, 'Minus_One_Mode': minus_one_mode,
        'Aligned_Entry_IDs': merged['__RiceSampleID__'].tolist(),
        'Marker_Selection': f'All SNP columns in {geno.name}; no extra SNP filtering or truncation',
        'Row_Order_Phenotype_Opt_In': False,
    }
    print(f'[DATA] {trait} samples={len(X)} SNPs={X.shape[1]:,}; '
          f'raw -1 count={minus_one_count:,} ({minus_one_mode}); '
          f'missing genotype cells={n_missing:,}; dropped missing phenotypes={before-len(merged)}', flush=True)
    print(f'[DATA] {trait} first SNPs: {snp_cols[:3]}, last SNPs: {snp_cols[-3:]}', flush=True)
    print(f'[DATA] {trait} y mean={float(y.mean()):.5f}, y SD={float(y.std()):.5f}', flush=True)
    if minus_one_count > 0:
        print('[IMPORTANT] Early PerfVAE.py treats -1 as MISSING. '
              'If -1 is a true genotype state, change CONFIG[minus_one_mode] to valid and rerun.', flush=True)
    return np.ascontiguousarray(X), np.ascontiguousarray(y), meta


# ------------------------ GPU-resident mini-batch execution --------------------------
def to_resident_tensors(xtr, ytr, xte, yte, device):
    """One host-to-GPU transfer for train/test partitions per fold."""
    def tensor(arr):
        return torch.as_tensor(np.ascontiguousarray(arr, dtype=np.float32), device=device)
    return tensor(xtr), tensor(ytr), tensor(xte), tensor(yte)


def batch_slices(X,y,batch_size,shuffle,device):
    n=X.shape[0]
    idx=torch.randperm(n,device=device) if shuffle else torch.arange(n,device=device)
    if shuffle and n%batch_size==1:
        idx=idx[:-1] # BatchNorm cannot train on batch of one
    for offset in range(0,len(idx),batch_size):
        sub=idx[offset:offset+batch_size]
        yield X[sub],y[sub]


def train_epoch_resident(model,loss_fn,X,y,optimizer,scheduler,batch_size,device):
    model.train()
    for bx,by in batch_slices(X,y,batch_size,True,device):
        optimizer.zero_grad(set_to_none=True)
        xr,yp,mu,lv,_=model(bx)
        loss=loss_fn(xr,bx,yp,by,mu,lv)
        # Avoid per-batch GPU synchronization. Check finite val loss per epoch.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
        optimizer.step(); scheduler.step()


@torch.inference_mode()
def evaluate_resident(model,loss_fn,X,y,batch_size,metrics=False):
    model.eval()
    total=torch.zeros((),device=X.device)
    pred=[]; target=[]
    n_batches=0
    for bx,by in batch_slices(X,y,batch_size,False,X.device):
        xr,yp,mu,lv,_=model(bx)
        total+=loss_fn(xr,bx,yp,by,mu,lv)
        n_batches+=1
        if metrics:
            pred.append(yp.detach().flatten()); target.append(by.flatten())
    mean_loss=(total/n_batches).item()
    if not metrics:
        return mean_loss
    p=torch.cat(pred); t=torch.cat(target)
    # Pearson invariant to affine transform with positive scaling.
    pc=p-p.mean(); tc=t-t.mean()
    div=(pc.square().sum()*tc.square().sum()).sqrt()
    pearson=(pc*tc).sum()/div.clamp_min(1e-12)
    ss_err=(p-t).square().sum(); ss_tot=tc.square().sum()
    r2=1-ss_err/ss_tot.clamp_min(1e-12)
    return mean_loss,float(pearson.item()),float(r2.item())


@torch.inference_mode()
def predict_resident(model, X, y, batch_size):
    """Return standardized predictions/targets; caller inverse-transforms predictions."""
    model.eval()
    pred, target = [], []
    for bx, by in batch_slices(X, y, batch_size, False, X.device):
        _, yp, _, _, _ = model(bx)
        pred.append(yp.detach().reshape(-1).cpu().numpy())
        target.append(by.detach().reshape(-1).cpu().numpy())
    return np.concatenate(pred), np.concatenate(target)


@torch.inference_mode()
def chunk_weight_diagnostics_resident(model,X,batch_size,export_path=None):
    encoder=model.snp_performer
    if not encoder.adaptive_weighting:return np.nan,np.nan
    cnt=0; ssum=0.; esum=0.; wsum=None
    for pos in range(0,X.shape[0],batch_size):
        w=encoder.get_chunk_weights(X[pos:pos+batch_size]).squeeze(-1).float()
        probs=w/encoder.num_chunks
        ssum+=w.std(dim=1,unbiased=False).sum().item()
        ent=-(probs*probs.clamp_min(1e-12).log()).sum(dim=1)
        if encoder.num_chunks>1: ent/=math.log(encoder.num_chunks)
        esum+=ent.sum().item(); cnt+=len(w)
        if export_path is not None:
            mean=w.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            wsum=mean if wsum is None else wsum+mean
    if export_path is not None:
        C,D,N=encoder.chunk_size,encoder.snp_dim,encoder.num_chunks
        result=pd.DataFrame({"Chunk_Index_1Based":np.arange(1,N+1),
                             "SNP_Start_1Based":np.arange(N)*C+1,
                             "SNP_End_1Based":np.minimum((np.arange(N)+1)*C,D),
                             "Mean_Weight":wsum/cnt})
        export_path.parent.mkdir(parents=True,exist_ok=True)
        result.to_csv(export_path,index=False,encoding="utf-8-sig")
    return ssum/cnt,esum/cnt


def fit_imputer(X_train):
    try:
        return SimpleImputer(strategy="most_frequent", keep_empty_features=True).fit(X_train)
    except TypeError:
        # For older scikit-learn versions, empty SNP columns are an explicit error.
        if np.isnan(X_train).all(axis=0).any():
            raise ValueError("All-missing SNPs exist; update scikit-learn or remove these markers.")
        return SimpleImputer(strategy="most_frequent").fit(X_train)


def prepared_fold(X, y, outer_indices):
    """Train-only preprocessing, NO inner validation split and NO leakage.

    Outer KFold: train ~90%, held-out test ~10%.
    """
    train_idx, test_idx = outer_indices
    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError("Too few samples in training or test partition.")
    imputer = fit_imputer(X[train_idx])
    X_train = imputer.transform(X[train_idx])
    X_test = imputer.transform(X[test_idx])
    if X_train.shape[1] != X.shape[1]:
        raise ValueError("Imputer unexpectedly changed SNP dimension.")
    sx = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    sy = StandardScaler().fit(y[train_idx])
    return (
        sx.transform(X_train).astype(np.float32),
        sy.transform(y[train_idx]).astype(np.float32),
        sx.transform(X_test).astype(np.float32),
        sy.transform(y[test_idx]).astype(np.float32),
        train_idx, test_idx, sy,
    )


def config_signature(args, experiment, X, y, configs):
    params = {
        "code_version": "rice-all-seven-traits-adaptive-pos-fixed-C256-20261024-v1",
        "minus_one_mode": args.minus_one_mode,
        "phenotype_column_overrides": args.phenotype_column_overrides,
        "data_kind": "rice",
        "allow_phenotype_row_order": args.allow_phenotype_row_order,
        "geno_path": str(Path(experiment["geno"]).resolve()),
        "pheno_path": str(Path(experiment["pheno"]).resolve()),
        "genotype_mtime_ns": Path(experiment["geno"]).stat().st_mtime_ns,
        "phenotype_mtime_ns": Path(experiment["pheno"]).stat().st_mtime_ns,
        "shape": [*X.shape, *y.shape],
        "design": args.design, "configs": configs,
        "folds": args.folds, "seed": args.seed,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "early_stopping": False,
        "validation_split": False, "test_model": "final_epoch",
        "dropout": args.dropout, "lr": args.lr, "max_lr": args.max_lr,
        "weight_decay": args.weight_decay, "latent_dim": args.latent_dim,
        "performer_dim": args.performer_dim,
        "weight_hidden": args.weight_hidden,
        "weight_temperature": args.weight_temperature,
        "loss_weights": args.loss_weights,
        "bootstrap_replicates": args.bootstrap_replicates,
        "train_only_preprocessing": True,
    }
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def csv_write_atomic(df, target):
    temp = target.with_name(target.stem + ".tmp.csv")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, target)



METRIC_COLUMNS = [
    "Pearson", "R2", "RMSE", "MAE", "Bias", "AbsBias", "Spearman",
    "Selection_Accuracy_Top10_Higher", "Selection_Accuracy_Top20_Higher",
    "Selection_Accuracy_Top10_Lower", "Selection_Accuracy_Top20_Lower",
]

def calculate_prediction_metrics(y_true, y_pred):
    """Metrics on the original phenotype scale; residual = prediction - truth."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[valid], yp[valid]
    if len(yt) < 2:
        return {name: np.nan for name in METRIC_COLUMNS}

    residual = yp - yt
    yt_centered = yt - yt.mean()
    yp_centered = yp - yp.mean()
    denom = np.sqrt(np.sum(yt_centered ** 2) * np.sum(yp_centered ** 2))
    pearson = float(np.sum(yt_centered * yp_centered) / denom) if denom > 0 else np.nan
    total_ss = float(np.sum(yt_centered ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / total_ss) if total_ss > 0 else np.nan
    try:
        sr = spearmanr(yt, yp)
        spearman = float(sr.statistic if hasattr(sr, "statistic") else sr[0])
    except Exception:
        spearman = np.nan

    def selection_accuracy(frac, higher=True):
        n = len(yt)
        k = max(1, int(math.ceil(frac * n)))
        true_order = np.argsort(yt, kind="mergesort")
        pred_order = np.argsort(yp, kind="mergesort")
        true_selected = true_order[-k:] if higher else true_order[:k]
        pred_selected = pred_order[-k:] if higher else pred_order[:k]
        return float(len(set(true_selected.tolist()) & set(pred_selected.tolist())) / k)

    bias = float(np.mean(residual))
    return {
        "Pearson": pearson,
        "R2": r2,
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "MAE": float(np.mean(np.abs(residual))),
        "Bias": bias,
        "AbsBias": abs(bias),
        "Spearman": spearman,
        "Selection_Accuracy_Top10_Higher": selection_accuracy(0.10, True),
        "Selection_Accuracy_Top20_Higher": selection_accuracy(0.20, True),
        "Selection_Accuracy_Top10_Lower": selection_accuracy(0.10, False),
        "Selection_Accuracy_Top20_Lower": selection_accuracy(0.20, False),
    }


def mean_t_ci(values, confidence=0.95):
    """Descriptive t interval across folds; CV folds are not independent replicates."""
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    if len(vals) < 2:
        return mean, sd, np.nan, np.nan
    critical = float(student_t.ppf((1 + confidence) / 2, df=len(vals) - 1))
    half = critical * sd / np.sqrt(len(vals))
    return mean, sd, float(mean - half), float(mean + half)


def bootstrap_oof_ci(y_true, y_pred, n_boot=1000, seed=20261024):
    """Percentile bootstrap CIs for pooled OOF metrics; interpret as sample-level CIs."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = len(yt)
    base = calculate_prediction_metrics(yt, yp)
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in METRIC_COLUMNS}
    if n < 3 or n_boot < 1:
        return base, {m: (np.nan, np.nan) for m in METRIC_COLUMNS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mvals = calculate_prediction_metrics(yt[idx], yp[idx])
        for metric in METRIC_COLUMNS:
            if np.isfinite(mvals[metric]):
                samples[metric].append(mvals[metric])
    cis = {}
    for metric, vals in samples.items():
        if len(vals) >= max(20, int(0.5 * n_boot)):
            cis[metric] = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
        else:
            cis[metric] = (np.nan, np.nan)
    return base, cis


def make_summary(folds_df, experiment):
    rows = []
    if folds_df.empty:
        return pd.DataFrame()
    for (variant, C), g in folds_df.groupby(["Variant", "Chunk_Size_C"], sort=False):
        row = {
            "Dataset": experiment["dataset"], "Trait": experiment["trait"],
            "Variant": variant, "Chunk_Size_C": int(C), "Completed_Folds": len(g),
            "Adaptive_Weighting": VARIANTS[variant][0],
            "Learnable_Positional_Encoding": VARIANTS[variant][1],
        }
        for col in METRIC_COLUMNS + [
            "Train_Pearson", "Train_R2", "Training_Time_s", "Epoch_Time_mean_s",
            "Peak_GPU_Memory_MB", "Chunk_Weight_Std", "Chunk_Weight_Entropy",
        ]:
            mean, sd, ci_low, ci_high = mean_t_ci(g[col])
            row[col + "_mean"] = mean
            row[col + "_std"] = sd
            row[col + "_CI95_lower"] = ci_low
            row[col + "_CI95_upper"] = ci_high
        for col in ("SNP_Dimension_D", "Num_Chunks_N", "Padding_P",
                    "Total_Params_M", "Trainable_Params_M"):
            row[col] = g.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Variant", "Chunk_Size_C"])


def make_oof_summary(oof_df, n_boot=1000):
    rows = []
    if oof_df.empty:
        return pd.DataFrame()
    for (dataset, trait, variant, C), g in oof_df.groupby(
            ["Dataset", "Trait", "Variant", "Chunk_Size_C"], sort=False):
        base, cis = bootstrap_oof_ci(
            g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
            n_boot=n_boot,
            seed=20261024 + int(C) + sum(ord(ch) for ch in str(trait)),
        )
        row = {
            "Dataset": dataset, "Trait": trait, "Variant": variant,
            "Chunk_Size_C": int(C), "OOF_Sample_Count": int(g["Sample_ID"].nunique()),
            "Bootstrap_Replicates": int(n_boot),
            "CI_Method": "Percentile bootstrap over pooled out-of-fold samples",
        }
        for metric in METRIC_COLUMNS:
            row[metric] = base[metric]
            row[metric + "_CI95_lower"] = cis[metric][0]
            row[metric + "_CI95_upper"] = cis[metric][1]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Trait", "Variant", "Chunk_Size_C"])


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


def make_paired_comparisons(folds_df):
    """Paired fold-level comparisons across C values; empty for the fixed-C run."""
    columns = [
        "Variant", "Metric", "Chunk_A", "Chunk_B", "Difference_Definition",
        "Paired_Folds", "Mean_Difference_A_minus_B", "Difference_SD",
        "Difference_CI95_lower", "Difference_CI95_upper", "Wilcoxon_p",
        "Inference_Note", "Holm_adjusted_p_within_metric",
    ]
    records = []
    if folds_df.empty or folds_df["Chunk_Size_C"].nunique() < 2:
        return pd.DataFrame(columns=columns)
    for metric in METRIC_COLUMNS:
        metric_records = []
        chunk_sizes = sorted(folds_df["Chunk_Size_C"].dropna().astype(int).unique())
        for i, c_a in enumerate(chunk_sizes):
            for c_b in chunk_sizes[i + 1:]:
                a = folds_df.loc[folds_df["Chunk_Size_C"].astype(int) == c_a,
                                 ["Fold", metric]].rename(columns={metric: "A"})
                b = folds_df.loc[folds_df["Chunk_Size_C"].astype(int) == c_b,
                                 ["Fold", metric]].rename(columns={metric: "B"})
                pair = a.merge(b, on="Fold", how="inner").dropna()
                if len(pair) < 3:
                    continue
                diff = pair["A"].to_numpy(dtype=float) - pair["B"].to_numpy(dtype=float)
                mean, sd, ci_low, ci_high = mean_t_ci(diff)
                try:
                    if np.allclose(diff, 0):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(pair["A"], pair["B"],
                                                 alternative="two-sided").pvalue)
                except Exception:
                    p_value = np.nan
                metric_records.append({
                    "Variant": "adaptive_pos", "Metric": metric,
                    "Chunk_A": int(c_a), "Chunk_B": int(c_b),
                    "Difference_Definition": "Metric at Chunk_A minus Metric at Chunk_B",
                    "Paired_Folds": int(len(pair)), "Mean_Difference_A_minus_B": mean,
                    "Difference_SD": sd, "Difference_CI95_lower": ci_low,
                    "Difference_CI95_upper": ci_high, "Wilcoxon_p": p_value,
                    "Inference_Note": "Exploratory: CV folds share overlapping training data.",
                })
        if metric_records:
            adjusted = holm_adjust([r["Wilcoxon_p"] for r in metric_records])
            for rec, p_adj in zip(metric_records, adjusted):
                rec["Holm_adjusted_p_within_metric"] = float(p_adj) if np.isfinite(p_adj) else np.nan
                records.append(rec)
    return pd.DataFrame(records)


def write_split_manifest(cv_splits, sample_ids, out_path):
    rows = []
    for fold_num, (_, test_idx) in enumerate(cv_splits, 1):
        for idx in test_idx:
            rows.append({
                "Sample_Index_0Based": int(idx),
                "Sample_ID": str(sample_ids[idx]),
                "Test_Fold": int(fold_num),
            })
    csv_write_atomic(pd.DataFrame(rows).sort_values("Sample_Index_0Based"), out_path)

def draw_plot(summary, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV files were saved, skipping figure.")
        return
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    for variant, g in summary.groupby("Variant", sort=False):
        g = g.sort_values("Chunk_Size_C")
        x = g["Chunk_Size_C"].to_numpy(dtype=int)
        y = g["Pearson_mean"].to_numpy(dtype=float)
        err = g["Pearson_std"].fillna(0).to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=err, fmt="o-", capsize=4, label=variant)
    ax.set_xscale("log", base=2)
    x = sorted(summary["Chunk_Size_C"].unique())
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Fixed SNP chunk size C (markers per chunk)")
    ax.set_ylabel("Test Pearson r (mean +/- SD over folds)")
    ax.set_title(f"{summary.iloc[0]['Dataset']} / {summary.iloc[0]['Trait']}")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_ablation_plot(summary, out_path, ablation_c):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    subset = summary[summary["Chunk_Size_C"] == ablation_c].copy()
    if subset.empty:
        return
    order = [mode for mode in VARIANTS if mode in set(subset["Variant"])]
    subset = subset.set_index("Variant").loc[order]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(order, subset["Pearson_mean"], yerr=subset["Pearson_std"].fillna(0),
           capsize=4)
    ax.set_ylabel("Test Pearson r (mean +/- SD over folds)")
    ax.set_xlabel(f"ASCM variant at C={ablation_c}")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_combined_plot(summary, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if summary.empty:
        return
    # For readability, only show sensitivity curves, not extra ablation points.
    subset = summary[summary["Variant"].isin(("original", "adaptive_pos"))]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for (dataset, trait, variant), g in subset.groupby(
            ["Dataset", "Trait", "Variant"], sort=False):
        g = g.sort_values("Chunk_Size_C")
        ax.errorbar(g["Chunk_Size_C"], g["Pearson_mean"],
                    yerr=g["Pearson_std"].fillna(0), marker="o", capsize=3,
                    label=f"{dataset}/{trait}: {variant}")
    ax.set_xscale("log", base=2)
    ticks = sorted(subset["Chunk_Size_C"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_xlabel("Fixed SNP chunk size C (markers)")
    ax.set_ylabel("Test Pearson r (mean +/- SD across folds)")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def get_weight_diagnostics(model, loader, device, export_path=None):
    """Evaluate learned chunk weights on held-out fold, not training samples.

    Exported weights are average chunk importances in that fold, NOT SNP causal
    effect estimates. Chunk indices are genomic INPUT COLUMN order (not bp).
    """
    encoder = model.snp_performer
    if not encoder.adaptive_weighting:
        return np.nan, np.nan
    model.eval()
    total_n, std_sum, entropy_sum = 0, 0.0, 0.0
    weight_sum = None
    for bx, _ in loader:
        w = encoder.get_chunk_weights(bx.to(device)).squeeze(-1).float()
        n = w.shape[0]
        probs = w / encoder.num_chunks
        std_sum += float(w.std(dim=1, unbiased=False).sum().cpu())
        h = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        if encoder.num_chunks > 1:
            h = h / math.log(encoder.num_chunks)
        entropy_sum += float(h.sum().cpu())
        total_n += n
        if export_path is not None:
            chunk_sum = w.sum(dim=0).cpu().numpy().astype(np.float64)
            weight_sum = chunk_sum if weight_sum is None else weight_sum + chunk_sum
    if export_path is not None:
        c, D, N = encoder.chunk_size, encoder.snp_dim, encoder.num_chunks
        means = weight_sum / total_n
        importance = pd.DataFrame({
            "Chunk_Index_1Based": np.arange(1, N + 1),
            "SNP_Start_1Based": np.arange(N) * c + 1,
            "SNP_End_1Based": np.minimum((np.arange(N) + 1) * c, D),
            "Mean_Weight": means,
        })
        export_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(export_path, index=False, encoding="utf-8-sig")
    return std_sum / total_n, entropy_sum / total_n



def run_experiment(args, exp):
    dataset, trait = exp["dataset"], exp["trait"]
    print(f"\nDATASET={dataset} | TRAIT={trait}", flush=True)
    X, y, data_meta = load_dataset(
        exp, allow_phenotype_row_order=args.allow_phenotype_row_order,
        verbose=args.verbose_data, minus_one_mode=args.minus_one_mode,
        phenotype_column_overrides=args.phenotype_column_overrides
    )
    if args.check_data_only:
        print(f"[CHECK DATA ONLY] Rice/{trait}: OK", flush=True)
        return None

    configs = make_config_grid(args)
    n, D = X.shape
    if n < args.folds * 3:
        raise ValueError(f"Only {n} samples: not enough for {args.folds}-fold cross-validation.")
    print(f"Configurations: {len(configs)} | samples={n} | SNPs={D:,}")
    if not configs or any(v != "adaptive_pos" for v, _ in configs):
        raise ValueError(f"This run must contain adaptive_pos only; found configs={configs}")

    output = Path(args.out_dir) / f"{safe_name(dataset)}_{safe_name(trait)}"
    output.mkdir(parents=True, exist_ok=True)
    fold_path = output / "fold_results.csv"
    summary_path = output / "summary.csv"
    oof_path = output / "oof_predictions.csv"
    oof_summary_path = output / "oof_metrics.csv"
    paired_path = output / "paired_comparisons.csv"
    split_path = output / "split_manifest.csv"
    signature = config_signature(args, exp, X, y, configs)
    metadata = {
        "Dataset": dataset, "Trait": trait, "SNP_Dimension": D,
        "Sample_Count": n, "Aligned_Sample_IDs": data_meta["Aligned_Entry_IDs"],
        "Chunk_Sizes": args.chunk_sizes, "Folds": args.folds,
        "Design": args.design, "Configurations": configs,
        "ASCM_Variant": "adaptive_pos only",
        "Signature": signature, "Genotype_Path": str(exp["geno"]),
        "Phenotype_Path": str(exp["pheno"]), "Data_Metadata": data_meta,
        "Early_Stopping": False, "Model_Selection": "final_epoch",
        "Epochs_Per_Fold": args.max_epochs,
        "Loss_Weights": args.loss_weights,
        "Learning_Rate_Scheduler": "OneCycleLR",
        "Learning_Rate_Peak": args.max_lr,
        "Weight_Decay": args.weight_decay,
        "Batch_Size": args.batch_size,
        "Metric_Definitions": {
            "Bias": "mean(y_pred - y_true), on the original phenotype scale",
            "AbsBias": "absolute value of fold-level Bias",
            "Selection_Accuracy": "overlap between predicted and observed top/bottom k; k=ceil(fraction*n_test)",
            "Fold_CI": "95% t interval across folds; descriptive because CV folds overlap",
            "OOF_CI": "95% percentile bootstrap interval across pooled out-of-fold samples",
            "Paired_Comparison": "paired Wilcoxon across folds, Holm-adjusted within each metric; exploratory",
        },
        "Note": "Strict sample-ID join; train-only imputation/scaling; 10-fold CV; fixed epochs; no early stopping. "
                "Only adaptive_pos at fixed C=256 is run. No within-run pairwise C/variant comparison is available; "
                "OOF predictions are retained for later comparisons against external baselines."
    }
    (output / "settings.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Same deterministic outer-fold assignments are shared by every chunk size.
    cv_splits = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X))
    sample_ids = data_meta["Aligned_Entry_IDs"]
    write_split_manifest(cv_splits, sample_ids, split_path)

    # Resume only folds for which BOTH fold metrics and sample-level OOF predictions exist.
    rows = []
    oof_rows = []
    existing_any = any(p.exists() for p in (fold_path, oof_path))
    if args.resume:
        if fold_path.exists():
            old = pd.read_csv(fold_path)
            if "Config_ID" not in old.columns or set(old["Config_ID"].astype(str)) != {signature}:
                raise ValueError(
                    f"Found incompatible fold_results.csv in {output}. "
                    "Use the new output directory or archive old results; do not mix configurations."
                )
            rows = old.to_dict("records")
        if oof_path.exists():
            old_oof = pd.read_csv(oof_path, dtype={"Sample_ID": str})
            if len(old_oof) and (
                "Config_ID" not in old_oof.columns
                or set(old_oof["Config_ID"].astype(str)) != {signature}
            ):
                raise ValueError(f"Found incompatible OOF predictions in {output}; do not mix configurations.")
            oof_rows = old_oof.to_dict("records")
        fold_keys = {(str(r["Variant"]), int(r["Chunk_Size_C"]), int(r["Fold"])) for r in rows}
        oof_keys = {(str(r["Variant"]), int(r["Chunk_Size_C"]), int(r["Fold"])) for r in oof_rows}
        valid_keys = fold_keys & oof_keys
        rows = [r for r in rows if (str(r["Variant"]), int(r["Chunk_Size_C"]), int(r["Fold"])) in valid_keys]
        oof_rows = [r for r in oof_rows if (str(r["Variant"]), int(r["Chunk_Size_C"]), int(r["Fold"])) in valid_keys]
    elif existing_any:
        raise FileExistsError(
            f"Results already exist in {output}. Set CONFIG['resume']=True or change CONFIG['out_dir']."
        )

    done = {(str(r["Variant"]), int(r["Chunk_Size_C"]), int(r["Fold"])) for r in rows}
    # Rewrite the paired files after dropping any incomplete/orphaned fold records.
    folds_df = pd.DataFrame(rows)
    oof_df = pd.DataFrame(oof_rows)
    if rows:
        folds_df = folds_df.sort_values(["Variant", "Chunk_Size_C", "Fold"])
        csv_write_atomic(folds_df, fold_path)
        csv_write_atomic(make_summary(folds_df, exp), summary_path)
    if oof_rows:
        oof_df = oof_df.sort_values(["Variant", "Chunk_Size_C", "Fold", "Sample_Index_0Based"])
        csv_write_atomic(oof_df, oof_path)

    device = torch.device(args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device, flush=True)

    for variant, C in configs:
        use_weights, use_pos = VARIANTS[variant]
        P, N = (-D) % C, (D + (-D) % C) // C
        print(f"\nVariant={variant} | C={C} | N={N} | Padding={P}", flush=True)
        for fold_num, outer_idx in enumerate(cv_splits, 1):
            if (variant, C, fold_num) in done:
                continue
            fold_seed = args.seed + fold_num
            seed_everything(fold_seed)
            xtr, ytr, xte, yte, tri, ti, y_scaler = prepared_fold(X, y, outer_idx)
            assert not (set(tri) & set(ti)), "Train/test overlap"
            Xtr, Ytr, Xte, Yte = to_resident_tensors(xtr, ytr, xte, yte, device)

            steps_per_epoch = len(Xtr) // args.batch_size + int(len(Xtr) % args.batch_size > 1)
            if steps_per_epoch < 1:
                raise ValueError("Training fold too small for BatchNorm with this batch size")
            model = EnhancedGeneticModel(
                snp_dim=D, chunk_size=C, latent_dim=args.latent_dim,
                performer_dim=args.performer_dim, dropout=args.dropout,
                adaptive_weighting=use_weights, learnable_position=use_pos,
                weight_hidden=args.weight_hidden, weight_temperature=args.weight_temperature
            ).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.max_lr, steps_per_epoch=steps_per_epoch,
                epochs=args.max_epochs
            )
            loss_fn = GeneticAwareLoss(args.loss_weights)
            epoch_times = []
            fold_start = time.perf_counter()
            iterator = tqdm(
                range(1, args.max_epochs + 1),
                desc=f"{variant} C={C} Fold={fold_num}", leave=False,
                disable=args.no_progress
            )
            for epoch in iterator:
                epoch_start = time.perf_counter()
                train_epoch_resident(
                    model, loss_fn, Xtr, Ytr, optimizer, scheduler, args.batch_size, device
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_times.append(time.perf_counter() - epoch_start)
                if args.epoch_log_interval > 0 and (
                    epoch == 1 or epoch % args.epoch_log_interval == 0 or epoch == args.max_epochs
                ):
                    print(f"  [{variant} C={C} fold={fold_num}/{args.folds}] "
                          f"epoch {epoch}/{args.max_epochs} completed", flush=True)

            _, train_r, train_r2 = evaluate_resident(
                model, loss_fn, Xtr, Ytr, args.batch_size, metrics=True
            )
            test_loss = evaluate_resident(
                model, loss_fn, Xte, Yte, args.batch_size, metrics=False
            )
            pred_scaled, _ = predict_resident(model, Xte, Yte, args.batch_size)
            y_pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
            y_true = y[ti].reshape(-1).astype(np.float64)
            test_metrics = calculate_prediction_metrics(y_true, y_pred)

            importance_path = (
                output / "chunk_importance" / f"{variant}_C{C}_Fold{fold_num}.csv"
                if args.export_chunk_weights and use_weights else None
            )
            weight_std, weight_entropy = chunk_weight_diagnostics_resident(
                model, Xte, args.batch_size, export_path=importance_path
            )
            if args.keep_checkpoints:
                checkpoint_dir = output / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"{variant}_C{C}_Fold{fold_num}_FinalEpoch.pth"
                )
            flush_gpu(device)
            training_time = time.perf_counter() - fold_start
            peak_mb = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else np.nan
            )

            rec = {
                "Config_ID": signature, "Dataset": dataset, "Trait": trait,
                "Reused_Fixed256": False, "Fold": fold_num, "Variant": variant,
                "Adaptive_Weighting": use_weights, "Learnable_Positional_Encoding": use_pos,
                "Chunk_Size_C": C, "SNP_Dimension_D": D,
                "Num_Chunks_N": N, "Padding_P": P,
                "Train_N": len(tri), "Test_N": len(ti),
                **test_metrics,
                "Test_Loss": test_loss, "Train_Pearson": train_r, "Train_R2": train_r2,
                "Chunk_Weight_Std": weight_std, "Chunk_Weight_Entropy": weight_entropy,
                "Model_Epoch": args.max_epochs, "Early_Stopped": False,
                "Epochs_Run": len(epoch_times), "Epoch_Time_mean_s": float(np.mean(epoch_times)),
                "Training_Time_s": training_time, "Peak_GPU_Memory_MB": peak_mb,
                "Total_Params_M": total_params / 1e6, "Trainable_Params_M": trainable_params / 1e6,
            }
            new_oof = []
            for local_pos, sample_idx in enumerate(ti):
                residual = float(y_pred[local_pos] - y_true[local_pos])
                new_oof.append({
                    "Config_ID": signature, "Dataset": dataset, "Trait": trait,
                    "Variant": variant, "Chunk_Size_C": int(C), "Fold": int(fold_num),
                    "Sample_Index_0Based": int(sample_idx), "Sample_ID": str(sample_ids[int(sample_idx)]),
                    "y_true": float(y_true[local_pos]), "y_pred": float(y_pred[local_pos]),
                    "Residual_pred_minus_true": residual, "Absolute_Error": abs(residual),
                    "Squared_Error": residual ** 2,
                })

            # Write OOF first, then fold metrics. If interrupted between writes, the orphan
            # OOF fold is ignored on resume because only folds present in both files are reused.
            oof_rows.extend(new_oof)
            oof_df = pd.DataFrame(oof_rows).sort_values(
                ["Variant", "Chunk_Size_C", "Fold", "Sample_Index_0Based"]
            )
            csv_write_atomic(oof_df, oof_path)
            rows.append(rec)
            done.add((variant, C, fold_num))
            folds_df = pd.DataFrame(rows).sort_values(["Variant", "Chunk_Size_C", "Fold"])
            csv_write_atomic(folds_df, fold_path)
            summary = make_summary(folds_df, exp)
            csv_write_atomic(summary, summary_path)

            print(
                f"  {variant} fold {fold_num}/{args.folds}: "
                f"r={test_metrics['Pearson']:.4f}, RMSE={test_metrics['RMSE']:.4f}, "
                f"MAE={test_metrics['MAE']:.4f}, Bias={test_metrics['Bias']:.4f}, "
                f"Spearman={test_metrics['Spearman']:.4f}",
                flush=True
            )
            del model, optimizer, scheduler, loss_fn, Xtr, Ytr, Xte, Yte
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Rebuild outputs after every chunk-size run, including resumed folds.
        folds_df = pd.DataFrame(rows)
        oof_df = pd.DataFrame(oof_rows)
        if not folds_df.empty:
            csv_write_atomic(make_summary(folds_df, exp), summary_path)
            csv_write_atomic(make_paired_comparisons(folds_df), paired_path)
        if not oof_df.empty:
            oof_summary = make_oof_summary(oof_df, n_boot=args.bootstrap_replicates)
            csv_write_atomic(oof_summary, oof_summary_path)
        if not folds_df.empty:
            entry = make_summary(folds_df, exp)
            entry = entry[(entry["Chunk_Size_C"] == C) & (entry["Variant"] == variant)]
            if len(entry):
                rr = entry.iloc[0]
                print(f"Summary {variant}, C={C}: Pearson={rr['Pearson_mean']:.4f} "
                      f"± {rr['Pearson_std']:.4f}; RMSE={rr['RMSE_mean']:.4f}; "
                      f"MAE={rr['MAE_mean']:.4f} ({int(rr['Completed_Folds'])}/{args.folds} folds)",
                      flush=True)

    folds_df = pd.DataFrame(rows)
    oof_df = pd.DataFrame(oof_rows)
    summary = make_summary(folds_df, exp)
    csv_write_atomic(summary, summary_path)
    if not oof_df.empty:
        csv_write_atomic(make_oof_summary(oof_df, n_boot=args.bootstrap_replicates), oof_summary_path)
        csv_write_atomic(make_paired_comparisons(folds_df), paired_path)
    if not args.no_plot and not summary.empty:
        draw_plot(summary, output / "Pearson_by_chunk_size.png")
    print(
        f"Saved: {fold_path}\nSaved: {summary_path}\nSaved: {oof_path}\n"
        f"Saved: {oof_summary_path}\nSaved: {paired_path}\nSaved: {split_path}",
        flush=True
    )
    return summary

def validate_config(args):
    if args.folds != 10 or args.max_epochs != 150:
        print("NOTICE: You changed 10-fold / 150-epoch fixed protocol in CONFIG.")
    if args.folds < 2 or args.max_epochs < 1 or args.batch_size < 2:
        raise ValueError("Need folds>=2, max_epochs>=1, batch_size>=2")
    if args.epoch_log_interval < 0:
        raise ValueError("epoch_log_interval must be >=0 (0 disables per-epoch logging)")
    if len(set(args.chunk_sizes)) != len(args.chunk_sizes) or any(
            C < 4 or C % 4 != 0 for C in args.chunk_sizes):
        raise ValueError("chunk_sizes must be unique positive multiples of 4")
    if args.weight_hidden < 1 or args.weight_temperature <= 0:
        raise ValueError("weight_hidden and weight_temperature must be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be >= 0")
    if not args.variants or set(args.variants) != {"adaptive_pos"}:
        raise ValueError("This experiment is intentionally restricted to adaptive_pos only.")
    if not isinstance(args.loss_weights, dict) or not {"pred", "recon", "kl"}.issubset(args.loss_weights):
        raise ValueError("CONFIG['loss_weights'] must contain pred, recon, and kl.")
    if args.design in ("efficient", "ablation") and args.ablation_c not in args.chunk_sizes:
        raise ValueError("ablation_c must be included in chunk_sizes")
    if args.design not in ("efficient", "full", "sensitivity", "ablation"):
        raise ValueError(f"Unknown experiment design: {args.design}")
    if args.variants is not None and not set(args.variants).issubset(VARIANTS):
        raise ValueError("Unknown ASCM variant")
    if args.device == "cuda" and not torch.cuda.is_available() and not args.check_data_only:
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")


def main():
    # All settings come from CONFIG above -- no command-line arguments.
    parser = argparse.ArgumentParser(description="Rice seven-trait fixed-C256 adaptive_pos metrics experiment")
    parser.add_argument('--check-data-only', action='store_true', help='Validate sample IDs / SNP counts without training')
    parser.add_argument('--data-dir', type=str, default=None, help='Override default script-relative data/ folder')
    parser.add_argument('--out-dir', type=str, default=None, help='Override results folder')
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default=None)
    cli = parser.parse_args()
    run_cfg = dict(CONFIG)
    if cli.check_data_only: run_cfg['check_data_only'] = True
    if cli.data_dir: run_cfg['data_dir'] = cli.data_dir
    if cli.out_dir: run_cfg['out_dir'] = cli.out_dir
    if cli.device: run_cfg['device'] = cli.device
    args = SimpleNamespace(**run_cfg)
    validate_config(args)
    seed_everything(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    print("=" * 78)
    print("PerfVAE ASCM | adaptive_pos ONLY | fixed epochs | no early stopping")
    print(f"Rice | Traits: {', '.join(args.traits)} | Chunks: {args.chunk_sizes}")
    print(f"10-fold CV | {args.max_epochs} epochs | no early stopping | Results: {args.out_dir}")
    print("=" * 78, flush=True)
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"Visible CUDA device 0: {torch.cuda.get_device_name(0)}", flush=True)
    experiments = discover_experiments(data_dir, args.traits)
    if not args.check_data_only and Performer is None:
        raise ImportError("Missing performer-pytorch; install it manually on your server")
    summaries = []
    for exp in experiments:
        summaries.append(run_experiment(args, exp))
    if args.check_data_only:
        print("All seven Rice phenotype files passed the available ID/phenotype/SNP checks. Training not started.")
        return
    all_df = pd.concat(summaries, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "Rice_AllTraits_C256_all_tasks_summary.csv"
    csv_write_atomic(all_df, all_path)
    if not args.no_plot:
        draw_combined_plot(all_df, out_dir / "Rice_AllTraits_C256_Pearson.png")
    print("\nAll-task summary:", all_path)
    print(all_df[["Dataset", "Trait", "Variant", "Chunk_Size_C", "Completed_Folds",
                  "Pearson_mean", "RMSE_mean", "MAE_mean", "Bias_mean",
                  "Spearman_mean", "Total_Params_M"]].to_string(index=False))


if __name__ == "__main__":
    main()
