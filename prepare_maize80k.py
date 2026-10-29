#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Maize1404: all-marker QC -> PLINK1.9 LD pruning -> deterministic physical-position 80k sampling.

Outputs a selected PLINK1 .bed/.bim/.fam file, an auditable marker manifest,
per-chromosome counts, and JSON parameters. Does NOT change the raw input.

Example:
 python 01_prepare_maize80k.py --bfile /data/cubic_1404_hmp2plink_maf0.02 \
   --out-dir /data/maize80k_qc_ld_v2 --plink plink --threads 6 --memory-mb 8192

Requirements: Python 3.9+, numpy, PLINK 1.9 in PATH (or --plink /path/to/plink).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
import numpy as np


def record(message):
    print(message, flush=True)


def execute(cmd, *, log):
    record('[RUN] ' + ' '.join(str(p) for p in cmd))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as handle:
        handle.write('COMMAND: ' + ' '.join(str(p) for p in cmd) + '\n')
        handle.flush()
        proc = subprocess.run([str(p) for p in cmd], stdout=handle, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'PLINK exited with status {proc.returncode}. Inspect {log}')


def read_fam_count(fam_path):
    with open(fam_path, encoding='utf-8') as f:
        return sum(bool(line.strip()) for line in f)


def inspect_bim(bim_path):
    n = 0
    with open(bim_path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): n += 1
    return n


def load_pruned_positions(bim_path, prune_file, allowed_chroms):
    # PLINK prune.in contains identifiers, not positions; marry them to QC .bim.
    with open(prune_file, encoding='utf-8') as f:
        keep = {line.strip() for line in f if line.strip()}
    if not keep:
        raise ValueError('PLINK .prune.in is empty; try a less aggressive LD threshold.')
    rows = defaultdict(list)
    seen = set()
    for line in open(bim_path, encoding='utf-8'):
        ss = line.split()
        if len(ss) < 6:
            raise ValueError('Malformed BIM line: ' + line[:100])
        chrom, marker_id, pos_txt = ss[0], ss[1], ss[3]
        if marker_id not in keep:
            continue
        if marker_id in seen:
            raise ValueError(f'Duplicate SNP ID in LD-retained BIM: {marker_id}. Resolve before --extract.')
        seen.add(marker_id)
        if chrom not in allowed_chroms:
            continue
        pos = int(pos_txt)
        if pos <= 0:
            continue
        rows[chrom].append((pos, marker_id))
    for chrom in rows:
        rows[chrom].sort(key=lambda x: (x[0], x[1]))
    eligible = sum(len(v) for v in rows.values())
    if eligible == 0:
        raise ValueError('No eligible autosomal SNPs with physical positions remain after pruning.')
    if len(seen) != len(keep):
        raise ValueError(f'{len(keep) - len(seen)} prune.in SNP IDs missing from the QC BIM.')
    return rows, len(keep)


def allocate_quotas(by_chrom, target, ordered_chroms):
    """Allocate marker counts proportional to each chromosome's observed physical span.

    Guarantee >=1 per represented chromosome and <= its marker capacity.
    """
    chroms = [c for c in ordered_chroms if c in by_chrom and by_chrom[c]]
    sizes = np.array([len(by_chrom[c]) for c in chroms], dtype=np.int64)
    if target > sizes.sum():
        raise ValueError(f'Only {sizes.sum():,} LD-pruned autosomal SNPs remain; cannot select {target:,}.')
    if target < len(chroms):
        raise ValueError('Target lower than represented chromosomes.')
    spans = np.array([max(1, by_chrom[c][-1][0] - by_chrom[c][0][0]) for c in chroms], dtype=float)
    allocations = np.ones(len(chroms), dtype=np.int64)
    remaining = int(target - allocations.sum())
    while remaining > 0:
        eligible = allocations < sizes
        weights = np.where(eligible, spans, 0.0)
        shares = remaining * weights / weights.sum()
        available = sizes - allocations
        add = np.minimum(np.floor(shares).astype(np.int64), available)
        progress = int(add.sum())
        if progress == 0:
            # Largest fractional remainder with fixed chromosome ordering as tiebreaker.
            j = int(np.argmax(np.where(eligible, shares, -1.0)))
            add[j] = 1
            progress = 1
        allocations += add
        remaining -= progress
    return dict(zip(chroms, (int(x) for x in allocations)))


def physical_positions_sample(rows, quota):
    """Monotone nearest-physical-grid positions with unique SNPs, exact quota.

    Targets divide the observed coordinate range at approximately equal BP gaps,
    NOT uniform genotype row indices. In sparse genomic regions exact BP spacing
    is impossible; report actual gaps as diagnostics instead of claiming equality.
    """
    n = len(rows)
    if quota == n:
        return rows.copy()
    positions = np.fromiter((p for p, _ in rows), count=n, dtype=np.int64)
    if quota == 1:
        mid = (int(positions[0]) + int(positions[-1])) / 2
        j = int(np.clip(np.searchsorted(positions, mid), 0, n - 1))
        return [rows[j]]
    targets = np.linspace(float(positions[0]), float(positions[-1]), quota)
    chosen = []
    last = -1
    for i, desired in enumerate(targets):
        minimum = last + 1
        maximum = n - (quota - i)
        j = int(np.searchsorted(positions, desired, side='left'))
        a = max(minimum, min(maximum, j))
        b = max(minimum, min(maximum, j - 1))
        chosen_idx = min((a, b), key=lambda t: (abs(float(positions[t]) - desired), t))
        chosen.append(rows[chosen_idx])
        last = chosen_idx
    assert len(set(marker for _, marker in chosen)) == quota
    return chosen


def sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as inp:
        for block in iter(lambda: inp.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    bfile = Path(args.bfile).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('.bed', '.bim', '.fam'):
        if not Path(str(bfile) + extension).is_file():
            raise FileNotFoundError(str(bfile) + extension)
    plink = shutil.which(args.plink) or args.plink
    if not Path(plink).is_file() and shutil.which(str(plink)) is None:
        raise FileNotFoundError(f'PLINK 1.9 executable not found: {args.plink}')
    version = subprocess.run([plink, '--version'], capture_output=True, text=True)
    version_txt = (version.stdout or version.stderr).strip().splitlines()
    record('PLINK version: ' + (' | '.join(version_txt[:2]) if version_txt else 'unknown'))
    if not ('PLINK v1.9' in ' '.join(version_txt) or 'PLINK 1.9' in ' '.join(version_txt) or 'PLINK v1.90' in ' '.join(version_txt)):
        record('[WARNING] This pipeline was written/tested for PLINK 1.9 CLI. Check your executable/version.')

    qc_prefix = out / '01_qc'
    ld_prefix = out / '02_ld'
    final_prefix = out / 'maize80k_qc_ld_physical'
    base = [plink, '--threads', str(args.threads), '--memory', str(args.memory_mb)]

    if not args.reuse_qc:
        # PLINK applies sample/SNP missingness and MAF thresholds to the original full panel.
        execute(base + ['--bfile', str(bfile), '--mind', str(args.mind), '--geno', str(args.geno),
                        '--maf', str(args.maf), '--make-bed', '--out', str(qc_prefix)],
                log=out / '01_qc_command.log')
    else:
        record('[REUSE] Reusing QC PLINK set (user assumes responsibility for matching QC thresholds).')
    for ext in ('.bed', '.bim', '.fam'):
        if not Path(str(qc_prefix) + ext).is_file():
            raise FileNotFoundError(str(qc_prefix) + ext)

    if not args.reuse_ld:
        execute(base + ['--bfile', str(qc_prefix), '--indep-pairwise', args.ld_window,
                        str(args.ld_step), str(args.ld_r2), '--out', str(ld_prefix)],
                log=out / '02_ld_command.log')
    else:
        record('[REUSE] Reusing LD list (user assumes responsibility for matching LD thresholds).')
    prune_file = Path(str(ld_prefix) + '.prune.in')
    if not prune_file.is_file():
        raise FileNotFoundError(prune_file)

    chroms = [str(c).strip() for c in args.chromosomes.split(',') if str(c).strip()]
    by_chrom, total_ld_pruned = load_pruned_positions(
        Path(str(qc_prefix) + '.bim'), prune_file, set(chroms))
    allocation = allocate_quotas(by_chrom, args.target, chroms)
    selected = {}
    for chrom, quota in allocation.items():
        selected[chrom] = physical_positions_sample(by_chrom[chrom], quota)
        record(f'Chr {chrom:>2}: LD retained {len(by_chrom[chrom]):>8,}, selected {quota:>7,}')
    ids = [marker for chrom in chroms for _, marker in selected.get(chrom, [])]
    assert len(ids) == args.target and len(set(ids)) == args.target

    extract = out / '03_selected_ids.txt'
    with extract.open('w', encoding='utf-8') as f:
        for marker in ids: f.write(marker + '\n')
    manifest = out / '03_selected_marker_manifest.tsv'
    with manifest.open('w', encoding='utf-8') as f:
        f.write('chrom\tposition_bp\tsnp_id\n')
        for chrom in chroms:
            for bp, marker in selected.get(chrom, []):
                f.write(f'{chrom}\t{bp}\t{marker}\n')

    stats_file = out / '03_chromosome_distribution.tsv'
    with stats_file.open('w', encoding='utf-8') as f:
        f.write('chrom\tld_retained\tselected\tmin_bp\tmax_bp\tmedian_selected_gap_bp\tmax_selected_gap_bp\n')
        for chrom in chroms:
            rows = selected.get(chrom, [])
            if not rows: continue
            pos = np.array([p for p, _ in rows], dtype=np.int64)
            gaps = np.diff(pos)
            f.write(f'{chrom}\t{len(by_chrom[chrom])}\t{len(rows)}\t{pos[0]}\t{pos[-1]}\t'
                    f'{float(np.median(gaps)) if len(gaps) else 0:.3f}\t'
                    f'{int(gaps.max()) if len(gaps) else 0}\n')

    execute(base + ['--bfile', str(qc_prefix), '--extract', str(extract), '--make-bed',
                    '--out', str(final_prefix)], log=out / '04_export_command.log')
    actual = inspect_bim(Path(str(final_prefix) + '.bim'))
    if actual != args.target:
        raise ValueError(f'Final PLINK subset has {actual} SNPs, expected {args.target}.')

    summary = {
        'input_plink_prefix': str(bfile), 'qc_plink_prefix': str(qc_prefix),
        'output_plink_prefix': str(final_prefix), 'original_variant_count': inspect_bim(Path(str(bfile)+'.bim')),
        'original_sample_count': read_fam_count(Path(str(bfile)+'.fam')),
        'qc_variant_count': inspect_bim(Path(str(qc_prefix)+'.bim')),
        'qc_sample_count': read_fam_count(Path(str(qc_prefix)+'.fam')),
        'ld_retained_total': total_ld_pruned, 'ld_retained_eligible_autosomal': sum(map(len, by_chrom.values())),
        'selected_variant_count': actual, 'selected_manifest_sha256': sha256_file(manifest),
        'method': 'PLINK1.9 QC; PLINK1.9 all-marker local LD pruning; per-chromosome BP-grid nearest-position deterministic subsampling',
        'allocation': allocation,
        'qc': {'maf':args.maf,'geno':args.geno,'mind':args.mind},
        'ld': {'physical_window': args.ld_window, 'step_in_variants':args.ld_step, 'r_squared_threshold':args.ld_r2},
        'chromosomes': chroms,
        'software': {'plink_version': version_txt, 'numpy_version': np.__version__, 'python':sys.version.split()[0]},
        'reuse_qc': bool(args.reuse_qc), 'reuse_ld': bool(args.reuse_ld),
    }
    with (out / '04_run_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    record(f'\nSUCCESS: {actual:,} SNPs -> {final_prefix}(.bed/.bim/.fam)')
    record(f'Manifest: {manifest}')
    record(f'Run summary: {out / "04_run_summary.json"}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bfile', required=True, help='Raw/input PLINK 1 BED/BIM/FAM prefix (omit extensions)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--plink', default='plink', help='PLINK 1.9 executable path')
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--memory-mb', type=int, default=8192)
    p.add_argument('--maf', type=float, default=0.05)
    p.add_argument('--geno', type=float, default=0.20, help='Max missing genotype fraction per SNP')
    p.add_argument('--mind', type=float, default=0.20, help='Max missing genotype fraction per sample')
    p.add_argument('--ld-window', default='50kb', help='PLINK LD physical window, e.g. 50kb')
    p.add_argument('--ld-step', type=int, default=5, help='Window advance in SNPs')
    p.add_argument('--ld-r2', type=float, default=0.8)
    p.add_argument('--target', type=int, default=80000)
    p.add_argument('--chromosomes', default='1,2,3,4,5,6,7,8,9,10')
    p.add_argument('--reuse-qc', action='store_true', help='Only if matching QC files already exist')
    p.add_argument('--reuse-ld', action='store_true', help='Only if matching LD results already exist')
    return p.parse_args()


if __name__ == '__main__':
    prepare(parse_args())
