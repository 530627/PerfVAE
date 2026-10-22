# -*- coding: utf-8 -*-
"""PerfVAE ASCM: fixed-150-epoch, 10-fold cross-validation on GPU.

Run without ANY command-line arguments:
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python -u \
      PerfVAE_ASCM_GPU_DataReady.py > gpu0.log 2>&1 &

Edit ONLY the CONFIG dictionary near the top to change experiments/parameters.
Default: Wheat2000 TKW, four ASCM variants, chunk size 256, 10 folds,
150 full training epochs per fold, NO early stopping and NO validation split.
Each fold trains on 90% of samples and tests on the held-out 10%.
The model at epoch 150 (NOT the best-validation checkpoint) is evaluated.

Genotype and phenotype files live in a data/ directory beside this script.
Preprocessing (imputation, X/Y scaling) is fitted using each training fold
only. All train/test tensors reside on the GPU during each fold, avoiding
repeated CPU/GPU mini-batch transfers. The code does not use the Internet.
New adaptive/positional modules are revisions to the submitted PerfVAE model.
"""

from __future__ import annotations

import math
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
import gc
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
    # Dataset names: "Wheat2000", "Rice", "Maize1404", "Wheat599".
    # Only Wheat2000/tkw is selected by default for the current formal run.
    "datasets": ["Wheat2000"],
    "traits": ["tkw"],
    "all_traits": False,
    "data_dir": str(DATA_DIR_DEFAULT),
    "max_maize_markers": 80000,   # Applies only to Maize1404.

    # Four-way ASCM ablation for one chunk size; other designs: efficient,
    # sensitivity, full. E.g., design="efficient", chunk_sizes=[32,64,128,256,512].
    "design": "ablation",
    "chunk_sizes": [256],
    "ablation_c": 256,
    "variants": None,     # None => all variants supported by the selected design.
    "weight_hidden": 64,
    "weight_temperature": 1.0,
    "export_chunk_weights": True,

    # FIXED experiment protocol: 10 folds x 150 epochs. No early stop,
    # no inner validation split. The FINAL epoch model is evaluated.
    "folds": 10,
    "max_epochs": 150,
    "batch_size": 64,
    "seed": 42,
    "dropout": 0.4,
    "lr": 1e-5,
    "max_lr": 5e-4,
    "weight_decay": 1e-5,
    "latent_dim": 128,
    "performer_dim": 256,
    "device": "cuda",     # CUDA_VISIBLE_DEVICES=0 makes this physical GPU 0.

    # Results stored separately from previous early-stopping experiments.
    "out_dir": str(SCRIPT_DIR / "ASCM_results_Fixed150_NoEarlyStop"),
    "resume": True,              # Skip COMPLETED folds only when config matches.
    "keep_checkpoints": False,   # True saves the FINAL epoch model per fold.
    "no_progress": True,        # Keep nohup logs readable.
    "epoch_log_interval": 10,   # Log progress without GPU->CPU data transfers.
    "no_plot": False,
    "genotype_first_col_is_snp": False, # Compatibility with optional generic loader.
    "allow_truncate": False,
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
    """Identical effective weights and reduction to the submitted code."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x_recon, x, y_pred, y, mu, logvar, aux_logits=None, aux_targets=None):
        pred = self.mse(y_pred, y)
        recon = self.mse(x_recon, x)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        aux = torch.tensor(0.0, device=x.device)
        if aux_logits is not None and aux_targets is not None:
            aux = self.ce(aux_logits, aux_targets.long())
        return 1.5 * pred + recon + 0.01 * kl + 0.0 * aux


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
    "original": (False, False),
    "adaptive": (True, False),
    "positional": (False, True),
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
            nn.Linear(snp_dim, 4086), nn.BatchNorm1d(4086), nn.GELU(),
            nn.Linear(4086, 1024), nn.BatchNorm1d(1024), nn.GELU(),
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


def load_data(geno_path, pheno_path, csv_skip_first_column=True, allow_truncate=False):
    path = Path(geno_path)
    if not path.is_file():
        raise FileNotFoundError(f"Genotype file not found: {path}")
    if not Path(pheno_path).is_file():
        raise FileNotFoundError(f"Phenotype file not found: {pheno_path}")

    if path.suffix.lower() in {".pkl", ".pickle"}:
        geno_obj = pd.read_pickle(path)
        X = geno_obj.to_numpy() if hasattr(geno_obj, "to_numpy") else np.asarray(geno_obj)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
        if csv_skip_first_column:
            if df.shape[1] < 2:
                raise ValueError("CSV genotype must have ID column + >=1 SNP column.")
            df = df.iloc[:, 1:]
        X = df.to_numpy()

    # Matches the submitted per-line phenotype parsing, but refuses silent truncation.
    y_values = []
    with open(pheno_path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.strip().replace(",", " ").replace("\t", " ").split()
            if not fields:
                continue
            try:
                y_values.append(float(fields[0]))
            except ValueError:
                # Original loader skips nonnumeric lines (usually a header).
                continue

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError("Genotype matrix and phenotype vector must be two-dimensional.")
    if len(X) != len(y):
        msg = (f"Sample count mismatch: genotype={len(X)}, phenotype={len(y)}. "
               "Check that sample IDs and row order match! The original script silently "
               "truncated rows, which can invalidate analyses.")
        if not allow_truncate:
            raise ValueError(msg + " Set CONFIG[allow_truncate]=True only after verifying row alignment.")
        print("WARNING:", msg, "Truncating to shorter length as explicitly requested.")
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
    if X.shape[1] < 1 or len(X) < 4:
        raise ValueError(f"Unexpected genotype shape: {X.shape}")
    if not np.isfinite(y).all():
        raise ValueError("Phenotype contains NaN/Inf. Align and clean phenotype rows first.")
    if np.isinf(X).any():
        raise ValueError("Genotype contains +/-Inf. Clean genotype file first.")
    return X, y



# ----------------- Dataset-specific readers grounded in related benchmark code -----------------
WHEAT_TRAITS = {
    "tkw": "2000_1_phe_tkw.txt", "testw": "2000_2_phe_testw.txt",
    "length": "2000_3_phe_length.txt", "width": "2000_4_phe_width.txt",
    "hard": "2000_5_phe_Hard.txt", "Prot": "2000_6_phe_Prot.txt",
}
RICE_TRAITS = {t: f"rice299_{t}.csv" for t in ("YLD", "PH", "TILLER", "FL", "LDG", "MAT", "Kg_ha")}
MAIZE_TRAITS = ("PH", "DTA", "KNPE", "KWPE")


def locate_file(data_dir, name):
    """Find exact filename in data/ and optional data/<dataset>/; reject ambiguity."""
    candidates = [x for x in data_dir.rglob("*") if x.is_file() and x.name.lower() == name.lower()]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"More than one file named {name!r}: {[str(p) for p in candidates]}")
    return candidates[0]


def discover_experiments(data_dir, dataset_filter=None, all_traits=False, trait_filter=None):
    """Identify task files according to the validated loaders in the user's Library.

    Representative default: Wheat2000/tkw, Rice/YLD, Maize1404/PH, Wheat599/Env1.
    Wheat599 requires vmgp_wheat599_X.pkl.csv, which was not in the screenshot.
    """
    allowed = {s.lower() for s in dataset_filter} if dataset_filter else None
    wanted = {s.lower() for s in trait_filter} if trait_filter else None
    def accept(ds, trait):
        return (allowed is None or ds.lower() in allowed) and (wanted is None or trait.lower() in wanted)

    tasks = []
    missing = []
    def add(ds, trait, geno, pheno, kind):
        if not accept(ds, trait):
            return
        gp = locate_file(data_dir, geno)
        pp = locate_file(data_dir, pheno)
        if gp is not None and pp is not None:
            tasks.append({"dataset": ds, "trait": trait, "geno": str(gp),
                          "pheno": str(pp), "kind": kind})
        else:
            missing.append(f"{ds}/{trait}: " + ", ".join(
                [f"genotype {geno}" if gp is None else "", f"phenotype {pheno}" if pp is None else ""]).strip(', '))

    for trait, name in WHEAT_TRAITS.items():
        if all_traits or trait == "tkw" or (wanted and trait.lower() in wanted):
            add("Wheat2000", trait, "2000gene.pkl", name, "wheat2000")
    for trait, name in RICE_TRAITS.items():
        if all_traits or trait == "YLD" or (wanted and trait.lower() in wanted):
            add("Rice", trait, "rice_geno2.txt", name, "rice")
    for trait in MAIZE_TRAITS:
        if all_traits or trait == "PH" or (wanted and trait.lower() in wanted):
            add("Maize1404", trait, "cubic_1404_hmp2plink_maf0.02.bed",
                "trait.txt", "maize_plink")
    for i in range(1, 5):
        trait = f"Env{i}"
        if all_traits or i == 1 or (wanted and trait.lower() in wanted):
            add("Wheat599", trait, "vmgp_wheat599_X.pkl.csv", f"vmgp_wheat{i}.Y", "wheat599")
    if missing:
        print("Missing optional/expected data files:")
        for line in missing: print("  -", line)
    if not tasks:
        raise FileNotFoundError("No complete tasks found under data/. Check your filenames and CONFIG dataset/trait names.")
    for t in tasks: print(f"Task {t['dataset']}/{t['trait']}: {Path(t['geno']).name} + {Path(t['pheno']).name}")
    return tasks


def load_dataset(exp, max_maize_markers=80000):
    """Read and align SNP/phenotype data; SNP missingness is imputed IN the CV fold."""
    kind = exp.get("kind", "generic")
    geno, pheno, trait = Path(exp["geno"]), Path(exp["pheno"]), exp["trait"]
    meta = {"Data_Kind": kind}
    if kind == "rice":
        gd = pd.read_csv(geno, sep=None, engine="python")
        pd_ = pd.read_csv(pheno)
        if "Entry" not in gd or "Entry" not in pd_:
            raise ValueError("Rice genotype and phenotype MUST both contain Entry column")
        if trait not in pd_.columns:
            raise ValueError(f"Rice phenotype {pheno.name} missing trait column {trait}")
        if gd["Entry"].duplicated().any() or pd_["Entry"].duplicated().any():
            raise ValueError("Rice Entry IDs are not unique; cannot align one-to-one")
        cols = [c for c in gd.columns if c != "Entry"]
        gd[cols] = gd[cols].apply(pd.to_numeric, errors="coerce")
        merged = pd_.merge(gd, how="inner", on="Entry", sort=False, validate="one_to_one")
        merged[trait] = pd.to_numeric(merged[trait], errors="coerce")
        merged = merged.dropna(subset=[trait])
        X = merged[cols].to_numpy(np.float32)
        y = merged[trait].to_numpy(np.float32).reshape(-1,1)
        meta["Aligned_Samples"] = len(merged)
        meta["Marker_Selection"] = "All SNP columns in rice299_geno2.txt; -1 retained as genotype"
    elif kind == "wheat599":
        # Genotype saved CSV has no header; first column = sample row index.
        gd = pd.read_csv(geno, header=None)
        X = gd.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        pd_ = pd.read_csv(pheno)
        y = pd.to_numeric(pd_.iloc[:, 0], errors="coerce").to_numpy(np.float32).reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError(f"Wheat599 sample mismatch X={len(X)} y={len(y)}")
        good = np.isfinite(y[:,0]); X, y = X[good], y[good]
        meta["Marker_Selection"] = "All genotype columns except saved row-index column"
    elif kind == "wheat2000":
        X0 = pd.read_pickle(geno)
        X = np.asarray(X0.to_numpy() if hasattr(X0, "to_numpy") else X0, dtype=np.float32)
        # Same row-order convention as existing Wheat2000 training scripts.
        y_vals=[]
        with pheno.open(encoding="utf-8-sig") as fh:
            for raw in fh:
                line=raw.strip()
                if not line:
                    raise ValueError(f"Blank phenotype row in {pheno}; row order would be broken")
                try: y_vals.append(float(line.split()[0].replace(",", "")))
                except ValueError: continue # optional header
        y=np.asarray(y_vals,dtype=np.float32).reshape(-1,1)
        if len(X)!=len(y):
            raise ValueError(f"Wheat2000 sample mismatch X={len(X)} y={len(y)}; verify row order")
        if not np.isfinite(y).all():
            raise ValueError("Wheat2000 phenotype NaN/Inf")
        meta["Marker_Selection"] = "All SNPs in 2000gene.pkl, row-order phenotype"
    elif kind == "maize_plink":
        prefix = geno.with_suffix("")
        bed,bim,fam=[Path(str(prefix)+ext) for ext in (".bed", ".bim", ".fam")]
        for f in (bed,bim,fam):
            if not f.is_file(): raise FileNotFoundError(f)
        n_samples=sum(1 for line in fam.open("rb") if line.strip())
        n_variants=sum(1 for line in bim.open("rb") if line.strip())
        with bed.open("rb") as fh: header=fh.read(3)
        if header != bytes((0x6c,0x1b,0x01)):
            raise ValueError(f"PLINK BED header not SNP-major: {header.hex()}")
        expected=3 + n_variants*((n_samples+3)//4)
        if bed.stat().st_size < expected:
            raise ValueError(f"PLINK BED truncated: expected {expected:,} bytes, found {bed.stat().st_size:,}")
        try:
            from pandas_plink import read_plink1_bin
        except ImportError as exc:
            raise ImportError("Maize1404 requires pandas-plink: pip install pandas-plink") from exc
        G=read_plink1_bin(str(bed),str(bim),str(fam),verbose=False)
        ids=np.asarray(G.sample.values).astype(str)
        total=int(G.variant.shape[0])
        if max_maize_markers < 1: raise ValueError("max-maize-markers must be >=1")
        if total > max_maize_markers:
            selected=np.unique(np.linspace(0,total-1,max_maize_markers,dtype=np.int64))
            G=G.isel(variant=selected)
        else:
            selected=np.arange(total,dtype=np.int64)
        X0=np.asarray(G.values,dtype=np.float32)
        if X0.shape[0]!=len(ids):
            raise ValueError(f"Unexpected PLINK orientation {X0.shape} vs IDs={len(ids)}")
        # pandas-plink encodes missing as -1. This is NOT the Rice genotype convention.
        X0[X0==-1]=np.nan
        traits=pd.read_csv(pheno,sep="\t",na_values=["NA","NaN","nan",""])
        if "<Trait>" not in traits.columns:
            raise ValueError("Maize trait.txt must contain the <Trait> sample-ID column")
        if trait not in traits.columns:
            raise ValueError(f"Trait {trait!r} not found in maize trait.txt: {traits.columns.tolist()}")
        traits["<Trait>"]=traits["<Trait>"].astype(str).str.strip()
        if traits["<Trait>"].duplicated().any():
            raise ValueError("Repeated maize phenotype ID; cannot align one-to-one")
        traits[trait]=pd.to_numeric(traits[trait],errors="coerce")
        fam_index=pd.DataFrame({"IID":ids,"PLINK_Row":np.arange(len(ids),dtype=np.int64)})
        joined=fam_index.merge(traits[["<Trait>",trait]],left_on="IID",right_on="<Trait>",
                               how="inner",sort=False,validate="one_to_one")
        joined=joined.dropna(subset=[trait]).reset_index(drop=True)
        if joined.empty: raise ValueError(f"No maize samples aligned for {trait}")
        rowidx=joined["PLINK_Row"].to_numpy(dtype=int)
        X=X0[rowidx].copy()
        y=joined[trait].to_numpy(np.float32).reshape(-1,1)
        meta.update({"Raw_SNPs": total, "Selected_SNPs": len(selected),
                     "Marker_Selection": "evenly spaced VARIANT INDICES, not physical-bp or LD pruning"})
        del G,X0
        gc.collect()
    else:
        X,y=load_data(geno,pheno)
        meta["Marker_Selection"]="generic loader"
    X=np.ascontiguousarray(X,dtype=np.float32)
    y=np.ascontiguousarray(y,dtype=np.float32)
    if X.ndim!=2 or y.shape!=(X.shape[0],1):
        raise ValueError(f"Invalid dataset shapes: X={X.shape}, y={y.shape}")
    if np.isinf(X).any() or not np.isfinite(y).all():
        raise ValueError("Inf in genotype or missing/Inf phenotype")
    print(f"Loaded {exp['dataset']}/{trait}: n={len(X)}, D={X.shape[1]}, missing SNP calls={np.isnan(X).sum():,}")
    return X, y, meta


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
        train_idx, test_idx,
    )


def config_signature(args, experiment, X, y, configs):
    params = {
        "code_version": "ascm-adaptive-gpu-fixed150-final-epoch-v3",
        "data_kind": experiment.get("kind", "generic"),
        "max_maize_markers": args.max_maize_markers,
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
        "genotype_first_col_is_snp": args.genotype_first_col_is_snp,
        "allow_truncate": args.allow_truncate,
        "train_only_preprocessing": True,
    }
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def csv_write_atomic(df, target):
    temp = target.with_name(target.stem + ".tmp.csv")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, target)


def make_summary(folds_df, experiment):
    rows = []
    for (variant, C), g in folds_df.groupby(["Variant", "Chunk_Size_C"], sort=False):
        row = {
            "Dataset": experiment["dataset"], "Trait": experiment["trait"],
            "Variant": variant, "Chunk_Size_C": int(C), "Completed_Folds": len(g),
            "Adaptive_Weighting": VARIANTS[variant][0],
            "Learnable_Positional_Encoding": VARIANTS[variant][1]
        }
        for col in (
            "Pearson", "R2", "Train_Pearson", "Train_R2",
            "Training_Time_s", "Epoch_Time_mean_s", "Peak_GPU_Memory_MB",
            "Chunk_Weight_Std", "Chunk_Weight_Entropy"
        ):
            series = pd.to_numeric(g[col], errors="coerce").dropna()
            row[col + "_mean"] = float(series.mean()) if len(series) else np.nan
            row[col + "_std"] = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        for col in ("SNP_Dimension_D", "Num_Chunks_N", "Padding_P",
                    "Total_Params_M", "Trainable_Params_M"):
            row[col] = g.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Variant", "Chunk_Size_C"])


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
    ax.set_xlabel("SNP chunk size C (number of markers per chunk)")
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
    ax.set_xlabel("SNP chunk size C (markers)")
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
    print(f"\n{'=' * 76}\nDATASET={dataset} | TRAIT={trait}\n{'=' * 76}", flush=True)
    X, y, data_meta = load_dataset(exp, max_maize_markers=args.max_maize_markers)
    configs = make_config_grid(args)
    n, D = X.shape
    if n < args.folds * 3:
        raise ValueError(f"Only {n} samples: not enough for {args.folds}-fold cross-validation.")
    print(f"Samples: {n} | SNP dimension: {D:,} | configurations: {len(configs)}")
    # The original first FC alone has D*4086 weights, independent of C.
    shortcut_gib = D * 4086 * 16 / (1024 ** 3)  # rough AdamW+gradient+weight bytes
    if shortcut_gib > 1.0:
        print(f"WARNING: Original Shortcut first FC requires ~{shortcut_gib:.2f} GiB "
              "for fp32 params+grads+AdamW states alone (excluding activations).")

    output = Path(args.out_dir) / f"{safe_name(dataset)}_{safe_name(trait)}"
    output.mkdir(parents=True, exist_ok=True)
    fold_path = output / "fold_results.csv"
    summary_path = output / "summary.csv"
    signature = config_signature(args, exp, X, y, configs)
    metadata = {"Dataset": dataset, "Trait": trait, "SNP_Dimension": D,
                "Sample_Count": n, "Chunk_Sizes": args.chunk_sizes, "Folds": args.folds,
                "Design": args.design, "Configurations": configs,
                "Signature": signature, "Genotype_Path": str(exp["geno"]),
                "Phenotype_Path": str(exp["pheno"]),
                "Data_Metadata": data_meta,
                "Note": "Train-only preprocessing on 90% outer-fold training data. "
                        "No validation split, no early stop; fixed 150 epochs and test final model. "
                        "C changes Performer dimension and parameter counts. "
                        "New weighted and positional variants are not the submitted model.",
                "Early_Stopping": False, "Model_Selection": "final_epoch",
                "Epochs_Per_Fold": args.max_epochs}
    (output / "settings.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.resume and fold_path.exists():
        folds_df = pd.read_csv(fold_path)
        sigs = set(folds_df["Config_ID"].astype(str)) if "Config_ID" in folds_df.columns else set()
        if sigs != {signature}:
            raise ValueError(f"Found incompatible fold_results.csv in {output}. "
                             "Change CONFIG[out_dir] or remove old results. "
                             "Never combine different configurations.")
        rows = folds_df.to_dict("records")
    else:
        if fold_path.exists():
            raise FileExistsError(f"Results already exist: {fold_path}. Set CONFIG[resume]=True "
                                  "(same configuration) or change CONFIG[out_dir].")
        rows = []

    done = {(str(r["Variant"]), int(r["Chunk_Size_C"]), int(r["Fold"])) for r in rows}
    cv_splits = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X))
    device = torch.device(args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    for variant, C in configs:
        use_weights, use_pos = VARIANTS[variant]
        P, N = (-D) % C, (D + (-D) % C) // C
        print(f"\nVariant={variant} | C={C} | N={N} | Padding={P}", flush=True)
        for fold_num, outer_idx in enumerate(cv_splits, 1):
            if (variant, C, fold_num) in done:
                print(f"  fold {fold_num}/{args.folds}: already complete; skip")
                continue
            fold_seed = args.seed + fold_num
            seed_everything(fold_seed)
            xtr, ytr, xte, yte, tri, ti = prepared_fold(X, y, outer_idx)
            assert not (set(tri) & set(ti)), "Train/test overlap"
            Xtr, Ytr, Xte, Yte = to_resident_tensors(
                xtr, ytr, xte, yte, device
            )
            # Drop only a singleton final training batch (BatchNorm requirement).
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.max_lr, steps_per_epoch=steps_per_epoch,
                epochs=args.max_epochs
            )
            loss_fn = GeneticAwareLoss()
            epoch_times = []
            fold_start = time.perf_counter()
            iterator = tqdm(range(1, args.max_epochs + 1),
                            desc=f"{variant} C={C} Fold={fold_num}", leave=False,
                            disable=args.no_progress)
            for epoch in iterator:
                epoch_start = time.perf_counter()
                train_epoch_resident(model, loss_fn, Xtr, Ytr, optimizer,
                                     scheduler, args.batch_size, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_times.append(time.perf_counter() - epoch_start)
                if (epoch == 1 or epoch % args.epoch_log_interval == 0
                        or epoch == args.max_epochs):
                    print(f"  [{variant} C={C} fold={fold_num}/{args.folds}] "
                          f"epoch {epoch}/{args.max_epochs} completed", flush=True)

            # NO early stopping, checkpoint selection, validation or weight rollback.
            # Predictions ALWAYS use the model trained through the last epoch.
            _, train_r, train_r2 = evaluate_resident(
                model, loss_fn, Xtr, Ytr, args.batch_size, metrics=True
            )
            test_loss, test_r, test_r2 = evaluate_resident(
                model, loss_fn, Xte, Yte, args.batch_size, metrics=True
            )
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
                torch.save(model.state_dict(),
                           checkpoint_dir / f"{variant}_C{C}_Fold{fold_num}_FinalEpoch.pth")
            flush_gpu(device)
            training_time = time.perf_counter() - fold_start
            peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                       if device.type == "cuda" else np.nan)

            rec = {
                "Config_ID": signature, "Dataset": dataset, "Trait": trait,
                "Fold": fold_num, "Variant": variant, "Adaptive_Weighting": use_weights,
                "Learnable_Positional_Encoding": use_pos,
                "Chunk_Size_C": C, "SNP_Dimension_D": D,
                "Num_Chunks_N": N, "Padding_P": P,
                "Train_N": len(tri), "Test_N": len(ti),
                "Pearson": test_r, "R2": test_r2, "Test_Loss": test_loss,
                "Train_Pearson": train_r, "Train_R2": train_r2,
                "Chunk_Weight_Std": weight_std,
                "Chunk_Weight_Entropy": weight_entropy,
                "Model_Epoch": args.max_epochs, "Early_Stopped": False,
                "Epochs_Run": len(epoch_times), "Epoch_Time_mean_s": np.mean(epoch_times),
                "Training_Time_s": training_time, "Peak_GPU_Memory_MB": peak_mb,
                "Total_Params_M": total_params / 1e6,
                "Trainable_Params_M": trainable_params / 1e6,
            }
            rows.append(rec)
            done.add((variant, C, fold_num))
            folds_df = pd.DataFrame(rows).sort_values(["Variant", "Chunk_Size_C", "Fold"])
            csv_write_atomic(folds_df, fold_path)
            summary = make_summary(folds_df, exp)
            csv_write_atomic(summary, summary_path)
            print(f"  {variant} fold {fold_num}/{args.folds}: test_r={test_r:.4f} "
                  f"R2={test_r2:.4f} final_epoch={args.max_epochs} "
                  f"params={total_params/1e6:.2f}M", flush=True)
            del model, optimizer, scheduler, loss_fn, Xtr,Ytr,Xte,Yte
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if rows:
            current = make_summary(pd.DataFrame(rows), exp)
            entry = current[(current["Chunk_Size_C"] == C) & (current["Variant"] == variant)]
            if len(entry):
                rr = entry.iloc[0]
                print(f"Summary {variant}, C={C}: Pearson={rr['Pearson_mean']:.4f} "
                      f"± {rr['Pearson_std']:.4f} "
                      f"({int(rr['Completed_Folds'])}/{args.folds} folds)", flush=True)

    summary = make_summary(pd.DataFrame(rows), exp)
    csv_write_atomic(summary, summary_path)
    if not args.no_plot:
        draw_plot(summary, output / "Pearson_by_chunk_size.png")
        draw_ablation_plot(summary, output / f"C{args.ablation_c}_Component_Ablation.png", args.ablation_c)
    print(f"Saved: {fold_path}\nSaved: {summary_path}")
    return summary


def validate_config(args):
    if args.folds != 10 or args.max_epochs != 150:
        print("NOTICE: You changed 10-fold / 150-epoch fixed protocol in CONFIG.")
    if args.folds < 2 or args.max_epochs < 1 or args.batch_size < 2:
        raise ValueError("Need folds>=2, max_epochs>=1, batch_size>=2")
    if args.epoch_log_interval < 1:
        raise ValueError("epoch_log_interval must be >=1")
    if len(set(args.chunk_sizes)) != len(args.chunk_sizes) or any(
            C < 4 or C % 4 != 0 for C in args.chunk_sizes):
        raise ValueError("chunk_sizes must be unique positive multiples of 4")
    if args.weight_hidden < 1 or args.weight_temperature <= 0:
        raise ValueError("weight_hidden and weight_temperature must be positive")
    if args.design in ("efficient", "ablation") and args.ablation_c not in args.chunk_sizes:
        raise ValueError("ablation_c must be included in chunk_sizes")
    if args.design not in ("efficient", "full", "sensitivity", "ablation"):
        raise ValueError(f"Unknown experiment design: {args.design}")
    if args.variants is not None and not set(args.variants).issubset(VARIANTS):
        raise ValueError("Unknown ASCM variant")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")


def main():
    # All settings come from CONFIG above -- no command-line arguments.
    args = SimpleNamespace(**CONFIG)
    validate_config(args)
    seed_everything(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    print("=" * 78)
    print("PerfVAE ASCM | fixed epochs, no early stopping, GPU-resident CV")
    print(f"Data: {data_dir}")
    print(f"Datasets: {args.datasets}; Traits: {args.traits}")
    print(f"Design: {args.design}; chunks: {args.chunk_sizes}")
    print(f"Folds: {args.folds}; epochs per fold: {args.max_epochs}; NO validation split")
    print(f"Results: {args.out_dir}")
    print("=" * 78, flush=True)
    if args.device == "cuda":
        print(f"Visible CUDA device 0: {torch.cuda.get_device_name(0)}", flush=True)
    experiments = discover_experiments(data_dir, args.datasets,
                                       all_traits=args.all_traits,
                                       trait_filter=args.traits)
    if Performer is None:
        raise ImportError("Missing performer-pytorch; install it manually on your server")
    summaries = []
    for exp in experiments:
        summaries.append(run_experiment(args, exp))
    all_df = pd.concat(summaries, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all_tasks_summary.csv"
    csv_write_atomic(all_df, all_path)
    if not args.no_plot:
        draw_combined_plot(all_df, out_dir / "All_Datasets_Pearson_by_chunk_size.png")
    print("\nAll-task summary:", all_path)
    print(all_df[["Dataset", "Trait", "Variant", "Chunk_Size_C", "Completed_Folds",
                  "Pearson_mean", "Pearson_std", "Total_Params_M"]].to_string(index=False))


if __name__ == "__main__":
    main()
