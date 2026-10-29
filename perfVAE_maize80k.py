#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the EXACT architecture/loss from the user's perfVAE0530.py on QC+LD+physical 80k PLINK data.

Input is the output of 01_prepare_maize80k.py; one separate 10-fold CV for each
trait (PH, KNPE, KWPE, DTA by default). Genotype preprocessing is fit only on
training subjects in each fold; trait records are aligned by explicit IID.

Typical use:
 python 02_train_perfVAE0530_maize80k.py \
   --bfile /data/maize80k_qc_ld_v2/maize80k_qc_ld_physical \
   --trait-file /data/trait.txt \
   --out-dir /data/maize80k_perfVAE0530_results --traits PH KNPE KWPE DTA

Deps: torch, performer-pytorch, pandas-plink, numpy, pandas, scikit-learn, scipy, tqdm.
This is a separate new experiment, not necessarily numerically comparable to
old 50k or 80k experiments that used a different SNP selection/model version.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset, random_split

# THIS is the ORIGINAL user-provided model, NOT the earlier perfVAE_LD.py model.
# Keep the copied file in the same folder; do not modify it to simplify data QC.
try:
    from perfVAE0530 import EnhancedGeneticModel, GeneticAwareLoss
except ImportError as exc:
    raise SystemExit('Missing dependency. Install: pip install performer-pytorch torch pandas-plink ' 
                     'numpy pandas scipy scikit-learn tqdm\n' + str(exc)) from exc
try:
    from pandas_plink import read_plink1_bin
except ImportError as exc:
    raise SystemExit('Missing pandas-plink: pip install pandas-plink\n' + str(exc)) from exc


def log(s):
    print(s, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_traits(path):
    """Read whitespace/tab/comma trait table; require an explicit identifier column."""
    with open(path, 'r', encoding='utf-8-sig') as fh:
        head = fh.readline()
    sep = '\t' if '\t' in head else (',' if ',' in head else r'\s+')
    df = pd.read_csv(path, sep=sep, engine='python', dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    if '<Trait>' in df.columns:
        id_col = '<Trait>'
    else:
        candidates = [c for c in df.columns if c.lower() in ('iid', 'id', 'sampleid', 'line', 'lineid', 'taxa')]
        if len(candidates) != 1:
            raise ValueError(f'Cannot identify phenotype IID column. Found: {df.columns.tolist()}')
        id_col = candidates[0]
    df[id_col] = df[id_col].astype(str).str.strip()
    if df[id_col].eq('').any() or df[id_col].duplicated().any():
        duplicates = df.loc[df[id_col].duplicated(), id_col].head(5).tolist()
        raise ValueError(f'Phenotype ID column must be nonempty and unique; duplicate examples={duplicates}')
    return df, id_col


def load_aligned(bfile, trait_file, trait, *, expected_markers=80000):
    """Keep only lines with measured trait and PLINK genotype; preserve PLINK sample ordering."""
    prefix = str(bfile)
    for suffix in ('.bed', '.bim', '.fam'):
        if not Path(prefix + suffix).is_file():
            raise FileNotFoundError(prefix + suffix)
    bim = pd.read_csv(prefix + '.bim', sep=r'\s+', header=None, usecols=[0, 1, 3], dtype={0: str, 1: str, 3: 'int64'})
    if len(bim) != expected_markers:
        raise ValueError(f'Expected {expected_markers:,} SNPs but BIM contains {len(bim):,}. Check stage 1.')
    fam = pd.read_csv(prefix + '.fam', sep=r'\s+', header=None, usecols=[0, 1], dtype=str)
    iid = fam.iloc[:, 1].astype(str).str.strip()
    if iid.duplicated().any():
        raise ValueError('PLINK .fam has duplicate IID values; use unique FID/IID join rather than IID-only matching.')
    pheno, id_col = get_traits(trait_file)
    if trait not in pheno.columns:
        lookup = {c.lower(): c for c in pheno.columns}
        if trait.lower() in lookup:
            trait = lookup[trait.lower()]
        else:
            raise ValueError(f'Trait {trait} not present. Columns: {pheno.columns.tolist()}')
    phenotype = pd.Series(pd.to_numeric(pheno[trait], errors='coerce').to_numpy(), index=pheno[id_col])
    match = iid.map(phenotype)
    valid = match.notna().to_numpy()
    rows = np.flatnonzero(valid)
    if len(rows) < 30:
        raise ValueError(f'Only {len(rows)} matched, nonmissing samples for {trait}; inspect IID format/trait data.')
    targets = match.iloc[rows].to_numpy(dtype=np.float32).reshape(-1, 1)
    log(f'[MATCH] {trait}: PLINK samples={len(iid)}, phenotype rows={len(pheno)}, '
        f'with measured matched trait={len(rows)}, SNPs={len(bim):,}')
    G = read_plink1_bin(prefix + '.bed', prefix + '.bim', prefix + '.fam', verbose=False)
    if len(G.sample) != len(iid) or len(G.variant) != len(bim):
        raise RuntimeError('pandas-plink sample/variant dimensions disagree with .fam/.bim.')
    # Verify ordering instead of silently assuming it when aligning phenotypes.
    g_ids = [str(s).strip() for s in G.sample.values]
    if g_ids != iid.tolist():
        raise RuntimeError('pandas-plink sample order does not match .fam IIDs; check FID/IID mapping.')
    # Read only the QC-pruned, 80k PLINK data, not the 11.8M parent file.
    X = np.asarray(G.isel(sample=rows).values, dtype=np.float32)
    if X.shape != (len(rows), expected_markers):
        raise RuntimeError(f'Unexpected genotype matrix shape: {X.shape}')
    X[~np.isfinite(X)] = np.nan
    del G
    gc.collect()
    return X, targets, [iid.iloc[i] for i in rows], bim, trait


def impute_train_test(X_tr, X_te, block=4096):
    """Most-frequent genotype (0,1,2) imputation using TRAIN fold only.

    Equivalent to sklearn SimpleImputer(strategy='most_frequent') on conventional
    PLINK dosages, except that all-missing training-fold markers are set to zero
    (and not silently removed). Avoids sklearn's many-column memory overhead.
    """
    p = X_tr.shape[1]
    for start in range(0, p, block):
        stop = min(p, start + block)
        a = X_tr[:, start:stop]
        b = X_te[:, start:stop]
        # Input PLINK encoding is allele dosage 0/1/2 or NaN.
        # np.argmax breaks frequency ties in favor of 0, as SimpleImputer does.
        counts = np.stack([(a == i).sum(axis=0) for i in (0, 1, 2)], axis=0)
        mode = counts.argmax(axis=0).astype(np.float32)
        r, c = np.where(np.isnan(a))
        a[r, c] = mode[c]
        r, c = np.where(np.isnan(b))
        b[r, c] = mode[c]
    return X_tr, X_te


def make_fold_tensors(X, y, tr_idx, te_idx):
    # Recompute from raw genotypes each fold, without test-derived imputation/scaling.
    X_tr = np.array(X[tr_idx], dtype=np.float32, copy=True)
    X_te = np.array(X[te_idx], dtype=np.float32, copy=True)
    X_tr, X_te = impute_train_test(X_tr, X_te)
    scaler_X, scaler_y = MinMaxScaler(feature_range=(-1, 1), copy=False), StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr)
    X_te = scaler_X.transform(X_te)
    y_tr = scaler_y.fit_transform(y[tr_idx])
    y_te = scaler_y.transform(y[te_idx])
    return (torch.from_numpy(np.ascontiguousarray(X_tr)), torch.from_numpy(np.ascontiguousarray(y_tr.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(X_te)), torch.from_numpy(np.ascontiguousarray(y_te.astype(np.float32))), scaler_y)


def loader_from_tensors(Xtr, ytr, Xte, yte, batch_size, seed):
    ds = TensorDataset(Xtr, ytr)
    n_tr = int(0.85 * len(ds))  # Original perfVAE0530.py: 85% training / 15% validation within fold
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_tr, len(ds) - n_tr], generator=generator)
    if len(train_ds) < 2 or len(val_ds) < 2:
        raise RuntimeError('Not enough samples for reliable training/validation evaluation.')
    # Drop singleton last batches because the original model has BatchNorm1d.
    effective_train_bs = min(batch_size, len(train_ds))
    if effective_train_bs < 2:
        raise RuntimeError('BatchNorm requires >=2 training samples per minibatch.')
    train_loader = DataLoader(train_ds, batch_size=effective_train_bs, shuffle=True,
                              drop_last=(len(train_ds) % effective_train_bs == 1),
                              generator=torch.Generator().manual_seed(seed + 123))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate(model, dataloader, scaler_y, loss_fn, device):
    model.eval()
    trues, preds, summed_loss = [], [], 0.0
    with torch.inference_mode():
        for bx, by in dataloader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            xrec, yp, mu, logvar, _ = model(bx)
            total, *_ = loss_fn(xrec, bx, yp, by, mu, logvar)
            summed_loss += float(total.detach()) * len(bx)
            trues.append(by.cpu().numpy())
            preds.append(yp.cpu().numpy())
    trues = scaler_y.inverse_transform(np.concatenate(trues)).ravel()
    preds = scaler_y.inverse_transform(np.concatenate(preds)).ravel()
    rval = float(np.corrcoef(trues, preds)[0, 1]) if np.std(preds) > 1e-8 and np.std(trues) > 1e-8 else float('nan')
    score = float(r2_score(trues, preds))
    return summed_loss / len(dataloader.dataset), rval, score, trues, preds


def state_on_cpu(model):
    # Original does deepcopy on GPU: avoid a second giant GPU model copy.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_one_trait(X, y, ids, trait, args):
    output = Path(args.out_dir) / trait
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('No CUDA GPU found. This huge original architecture is very slow on CPU. '
                           'Use --allow-cpu only for tiny tests.')
    log(f'\n[MODEL] Exact EnhancedGeneticModel + GeneticAwareLoss from perfVAE0530.py, '
        f'latent_dim=128, performer_dim=256, dropout={args.dropout}, chunk_size=256')
    log(f'[TRAIN] batch={args.batch_size}, epochs<={args.epochs}, lr={args.lr}, '
        f'OneCycleLR(max_lr={args.max_lr}), patience={args.patience}, seed={args.seed}, device={device}')
    log('[MEMORY] 80,000 x 4,086 first-layer weights alone use ~1.31 GB FP32; '
        'optimizer/gradients and other layers require several more GB.')
    if device.type == 'cuda':
        log(f'[GPU] {torch.cuda.get_device_name(0)}; '
            f'total VRAM {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB')
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        if args.fold and fold != args.fold:
            continue
        seed_everything(args.seed + fold)
        t0 = time.time()
        log(f'\n[{trait}] Fold {fold}/10: train={len(train_idx)}, test={len(test_idx)}')
        Xtr, ytr, Xte, yte, scaler_y = make_fold_tensors(X, y, train_idx, test_idx)
        train_loader, val_loader, test_loader = loader_from_tensors(
            Xtr, ytr, Xte, yte, args.batch_size, args.seed + fold)
        model = optimizer = scheduler = best_state = loss_fn = None
        try:
            model = EnhancedGeneticModel(snp_dim=X.shape[1], latent_dim=128,
                                         performer_dim=256, dropout=args.dropout).to(device)
            loss_fn = GeneticAwareLoss(pred_weight=1.5, recon_weight=1.0, aux_weight=0).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                            steps_per_epoch=len(train_loader), epochs=args.epochs)
            best_val = float('inf')
            best_state = None
            best_epoch = 0
            waits = 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                running = 0.0
                for bx, by in train_loader:
                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    # Original perfVAE0530 aux_weight=0. No auxiliary logits are
                    # needed for the loss; the exact model and gradients through
                    # the enabled components are otherwise unchanged.
                    xrec, yp, mu, logvar, _ = model(bx)
                    loss, *_ = loss_fn(xrec, bx, yp, by, mu, logvar)
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(f'Nonfinite loss in fold {fold}, epoch {epoch}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    running += float(loss.detach()) * len(bx)
                vl, vr, _, _, _ = evaluate(model, val_loader, scaler_y, loss_fn, device)
                if vl < best_val:
                    best_val, best_epoch, waits = vl, epoch, 0
                    # Keep checkpoint on CPU, not another >1 GiB copy on GPU.
                    if best_state is not None:
                        del best_state
                    best_state = state_on_cpu(model)
                    if args.save_checkpoints:
                        torch.save(best_state, output / f'fold_{fold}_best_state.pt')
                else:
                    waits += 1
                log(f'[{trait}] fold={fold} epoch={epoch:03d} '
                    f'train_loss={running / len(train_loader.dataset):.5f} '
                    f'val_loss={vl:.5f} val_r={vr:.4f} '
                    f'best_epoch={best_epoch} patience={waits}/{args.patience}')
                if waits >= args.patience:
                    break
            if best_state is None:
                raise RuntimeError('No valid checkpoint saved for this fold.')
            model.load_state_dict(best_state)
            test_loss, test_r, test_r2, y_true, y_pred = evaluate(model, test_loader, scaler_y, loss_fn, device)
            pd.DataFrame({'sample_id': [ids[i] for i in test_idx],
                          'observed': y_true, 'predicted': y_pred}).to_csv(
                              output / f'fold_{fold}_predictions.csv', index=False)
            row = {'trait':trait, 'fold':fold, 'n_train':len(train_idx), 'n_test':len(test_idx),
                   'best_epoch':best_epoch, 'val_loss':best_val, 'test_loss':test_loss,
                   'pearson_r':test_r, 'r2_score':test_r2, 'pearson_r_squared':test_r**2,
                   'seconds':round(time.time() - t0, 2)}
            fold_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(output / 'fold_results.csv', index=False)
            log(f'[RESULT] {trait} fold {fold}: r={test_r:.5f}, R2={test_r2:.5f}, best_epoch={best_epoch}')
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError('CUDA OOM with original perfVAE0530 architecture. '
                               'Try a smaller --batch-size (model unchanged) or larger GPU. '
                               'Do NOT silently replace the model with perfVAE_LD.py.') from exc
        finally:
            # Release the full >1 GiB model, optimizer and fold tensors explicitly.
            del model, optimizer, scheduler, best_state, loss_fn
            del train_loader, val_loader, test_loader
            del Xtr, ytr, Xte, yte
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if len(fold_rows) > 1:
        res = pd.DataFrame(fold_rows)
        summary = {'trait':trait, 'folds_completed':len(res), 'n_samples':len(ids),
                   'n_snps':X.shape[1],
                   'pearson_r_mean': float(res.pearson_r.mean()),
                   'pearson_r_sd': float(res.pearson_r.std(ddof=1)),
                   'r2_score_mean': float(res.r2_score.mean()),
                   'r2_score_sd': float(res.r2_score.std(ddof=1))}
        with (output / 'summary.json').open('w') as fh:
            json.dump(summary, fh, indent=2)
        log('[SUMMARY] ' + json.dumps(summary, indent=2))
    return fold_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bfile', required=True, help='Stage 1 output PLINK prefix (no .bed suffix)')
    parser.add_argument('--trait-file', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--traits', nargs='+', default=['PH','KNPE','KWPE','DTA'])
    parser.add_argument('--expected-markers', type=int, default=80000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64, help='64 in original perfVAE0530.py')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--fold', type=int, default=0, choices=range(0,11), metavar='{0..10}',
                        help='Run just one fold (1..10); 0 = all 10 folds')
    parser.add_argument('--save-checkpoints', action='store_true',
                        help='WARNING: ~1.4 GB per fold, up to ~14 GB per trait! Default is metrics/predictions only.')
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    # Self-document all values; config does not claim the submitted paper used this new SNP panel.
    import hashlib
    root = Path(__file__).parent
    source = root / 'perfVAE0530.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with open(Path(args.out_dir) / 'experiment_config.json','w', encoding='utf-8') as fh:
        data = vars(args).copy()
        data.update({'model_source': str(source.resolve()), 'model_source_sha256': source_hash,
                     'data_processing': 'global genotype-only QC/LD; 80k physical BP sampling; '
                        'training-fold-only missing genotype imputation and MinMax scaling',
                     'model_architecture': 'unmodified EnhancedGeneticModel from perfVAE0530.py',
                     'loss': 'unmodified GeneticAwareLoss from perfVAE0530.py: pred1.5/recon1.0/KL0.01/aux0'})
        json.dump(data, fh, indent=2, ensure_ascii=False)
    for trait in args.traits:
        X, y, ids, snps, resolved_trait = load_aligned(
             args.bfile, args.trait_file, trait, expected_markers=args.expected_markers)
        try:
            run_one_trait(X, y, ids, resolved_trait, args)
        finally:
            del X, y, ids, snps
            gc.collect()


if __name__ == '__main__':
    main()
