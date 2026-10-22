# PerfVAE ASCM

PyTorch implementation of a variational model for phenotype prediction from SNP genotypes, with optional sample-dependent chunk weighting and learnable chunk positional embeddings.

The training and evaluation workflow is implemented in [`PerfVAE.py`](PerfVAE.py). Experiment settings are controlled by the `CONFIG` dictionary near the top of that file.

By default, the script runs **Wheat2000 / `tkw`**, compares **four model variants** at chunk size **256**, and performs **10-fold cross-validation with 150 training epochs per fold**. This produces 40 fold-level training runs. Each run is evaluated using its final-epoch model.

## Repository layout

| Path | Purpose |
| --- | --- |
| `PerfVAE.py` | Model definitions, dataset readers, preprocessing, training, evaluation, and result export |
| `data/` | Local input directory, located beside the script by default |
| `ASCM_results_Fixed150_NoEarlyStop/` | Default output directory, generated during execution |

Dataset files are supplied locally by the user. The script does not download data.

## Model

`EnhancedGeneticModel` combines two genotype feature branches:

1. **Chunked Performer branch.** The SNP vector is zero-padded when necessary, reshaped into fixed-length chunks, and normalized with `LayerNorm`. Optional chunk weighting and positional embeddings are applied before a two-layer, four-head Performer. Mean and max pooling are concatenated and projected to a feature vector.
2. **Linear shortcut branch.** A multilayer network maps the full SNP vector through widths `4096 → 1024 → 256 → 64`, with batch normalization, GELU activations, and dropout as defined in the code.

The branch features are concatenated to estimate the latent mean and log variance. Training uses reparameterized latent samples; evaluation uses the latent mean. A prediction head estimates the phenotype, while a Performer-based decoder reconstructs the SNP input.

### Model variants

The variants change the encoder's chunk weighting and positional embeddings. The shortcut, latent heads, phenotype predictor, and decoder remain part of every variant.

| `Variant` | Sample-dependent chunk weights | Learnable chunk positional embeddings |
| --- | --- | --- |
| `original` | Disabled | Disabled |
| `adaptive` | Enabled | Disabled |
| `positional` | Disabled | Enabled |
| `adaptive_pos` | Enabled | Enabled |

For adaptive variants, a shared scoring network assigns weights independently for each sample:

$$
w_i = N\,\operatorname{softmax}_i(s/\tau),
$$

where $N$ is the number of chunks and $\tau$ is `weight_temperature`. Weights therefore sum to $N$ and have mean 1 across chunks. The scoring network's final layer is initialized to zero, giving uniform initial weights. Positional embeddings are also initialized to zero.

Chunk boundaries and lengths are fixed by input column order and `chunk_size`. Adaptive weighting learns chunk weights, rather than chunk boundaries. Positional embeddings represent chunk indices, rather than physical genomic coordinates.

### Training objective

The effective objective in `GeneticAwareLoss` is:

$$
\mathcal{L} =
1.5\,\operatorname{MSE}(\hat{y},y)
+\operatorname{MSE}(\hat{x},x)
+0.01\,\mathcal{L}_{\mathrm{KL}},
$$

with

$$
\mathcal{L}_{\mathrm{KL}} =+-\frac{1}{2B}\sum_{b=1}^{B}\sum_{j=1}^{L}
\left(1+\log\sigma_{bj}^{2}-\mu_{bj}^{2}-\sigma_{bj}^{2}\right).
$$

Here, $x$ and $y$ are the fold-preprocessed genotype and phenotype inputs, $B$ is the batch size, and $L$ is the latent dimension. Both MSE terms use PyTorch's mean reduction.

An auxiliary classifier is defined in the decoder, but the training loop does not pass auxiliary targets and its loss coefficient is zero.

## Dependencies

The script imports the following third-party packages:

```bash
pip install torch numpy pandas scikit-learn performer-pytorch tqdm
```

For figure generation:

```bash
pip install matplotlib
```

For Maize1404 PLINK input:

```bash
pip install pandas-plink
```

For the default CUDA configuration, the PyTorch installation must support the available GPU. Package versions are not pinned in the script. If Matplotlib is unavailable, the plotting functions skip figures and retain CSV outputs.

## Input data

Place prepared data under `data/` beside `PerfVAE.py`, or change `CONFIG["data_dir"]`. The script searches recursively, so dataset subfolders are supported. Filename matching is case-insensitive; duplicate matching filenames raise an error.

The table below lists the filenames and trait keys implemented in this version of the code. Dataset names and filenames do not enforce a fixed sample count; the loaded and aligned data determine the actual sample count and SNP dimension.

| Dataset | Genotype input | Phenotype input | Supported trait keys |
| --- | --- | --- | --- |
| `Wheat2000` | `2000gene.pkl` | See mapping below | `tkw`, `testw`, `length`, `width`, `hard`, `Prot` |
| `Rice` | `rice_geno2.txt` | `rice299_{trait}.csv` | `YLD`, `PH`, `TILLER`, `FL`, `LDG`, `MAT`, `Kg_ha` |
| `Maize1404` | `cubic_1404_hmp2plink_maf0.02.bed`, with matching `.bim` and `.fam` files | `trait.txt` | `PH`, `DTA`, `KNPE`, `KWPE` |
| `Wheat599` | `vmgp_wheat599_X.pkl.csv` | `vmgp_wheat1.Y` through `vmgp_wheat4.Y` | `Env1`, `Env2`, `Env3`, `Env4` |

### Wheat2000

`2000gene.pkl` must contain a numeric samples-by-SNPs matrix, stored as a pandas object or array. All columns are used as SNP features.

| Trait key | Phenotype filename |
| --- | --- |
| `tkw` | `2000_1_phe_tkw.txt` |
| `testw` | `2000_2_phe_testw.txt` |
| `length` | `2000_3_phe_length.txt` |
| `width` | `2000_4_phe_width.txt` |
| `hard` | `2000_5_phe_Hard.txt` |
| `Prot` | `2000_6_phe_Prot.txt` |

Each phenotype row supplies a numeric value in its first whitespace-separated field. Nonnumeric header lines are skipped; blank rows raise an error. Genotype and phenotype rows must already be in matching sample order, and their sample counts must agree. This reader does not match sample IDs.

### Rice

The genotype and phenotype tables must both contain a unique `Entry` column. Each phenotype file must also contain its corresponding trait column. Samples are inner-joined by `Entry`, and rows with missing or nonnumeric values for the selected phenotype are removed.

All genotype columns except `Entry` are used. Nonnumeric genotype entries are converted to missing values; numeric `-1` values are retained as genotype values by this reader.

For example, `rice299_YLD.csv` must contain `Entry` and `YLD` columns. The `299` in the filename is a naming convention, rather than a hard-coded sample limit.

### Maize1404

The reader requires a SNP-major PLINK BED file and matching BIM/FAM files. `trait.txt` must be tab-separated and include the sample-ID column `<Trait>` and the requested trait column. Phenotypes are matched to PLINK sample IDs; rows with missing selected phenotypes are removed. Genotype values encoded as `-1` are converted to missing values.

By default, at most 80,000 markers are retained using evenly spaced **variant indices**. This selection does not use physical base-pair distances, LD pruning, or phenotype associations.

### Wheat599

The genotype CSV is read with `header=None`; its first column is discarded as a saved row index. Remaining columns must contain numeric SNP values.

Each phenotype file is read with `pandas.read_csv` using its default header handling, and the first parsed column is used as the phenotype. The parsed genotype and phenotype row counts must match, and sample order must already agree. Rows with nonfinite phenotype values are removed. This reader does not match sample IDs.

## Run

Edit `CONFIG` in `PerfVAE.py`, then run:

```bash
python -u PerfVAE.py
```

For background execution on a Linux server using physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python -u PerfVAE.py > gpu0.log 2>&1 &
```

The script reads settings from `CONFIG` and does not parse command-line options.

### Default configuration

| Setting | Default |
| --- | --- |
| `datasets` / `traits` | `["Wheat2000"]` / `["tkw"]` |
| `all_traits` | `False` |
| `design` | `"ablation"` |
| `chunk_sizes` / `ablation_c` | `[256]` / `256` |
| `variants` | `None` — use all variants selected by the design |
| `folds` / `max_epochs` | `10` / `150` |
| `batch_size` / `seed` | `64` / `42` |
| `dropout` / `weight_decay` | `0.4` / `1e-5` |
| `lr` / `max_lr` | `1e-5` / `5e-4` |
| `latent_dim` / `performer_dim` | `128` / `256` |
| `weight_hidden` / `weight_temperature` | `64` / `1.0` |
| `device` | `"cuda"` |
| `resume` | `True` |
| `export_chunk_weights` | `True` |
| `keep_checkpoints` | `False` |
| `no_progress` / `epoch_log_interval` | `True` / `10` |
| `no_plot` | `False` |

Optimization uses AdamW, gradient clipping with maximum norm 1, and a per-batch `OneCycleLR` scheduler. `lr` is passed to AdamW; `OneCycleLR` subsequently controls the learning rate using `max_lr` and its own default scheduling parameters.

`performer_dim` controls the encoder output projection and decoder latent projection. The internal Performer dimension equals the chunk size, so changing chunk size also changes parameter counts and model capacity.

### Select datasets and traits

For Rice yield, change these entries in `CONFIG`:

```python
"datasets": ["Rice"],
"traits": ["YLD"],
"all_traits": False,
```

To discover all supported traits for all four datasets:

```python
"datasets": ["Wheat2000", "Rice", "Maize1404", "Wheat599"],
"traits": None,
"all_traits": True,
```

Clear `traits` when using `all_traits=True`; otherwise, the trait filter still restricts the selected tasks. Missing tasks are reported and skipped, while execution stops if no complete task is found. Dataset and trait selection is case-insensitive, but table column names must match those expected by the readers.

### Experiment designs

| `design` | Configurations generated before applying `variants` |
| --- | --- |
| `ablation` | All four variants at `ablation_c` |
| `sensitivity` | `original` and `adaptive_pos` at every listed chunk size |
| `efficient` | `original` and `adaptive_pos` at every listed chunk size, plus `adaptive` and `positional` at `ablation_c` |
| `full` | All four variants at every listed chunk size |

For example, a chunk-size sensitivity experiment can use:

```python
"design": "sensitivity",
"chunk_sizes": [32, 64, 128, 256, 512],
"variants": None,
```

Chunk sizes must be unique multiples of 4 and at least 4. For `ablation` and `efficient`, `ablation_c` must appear in `chunk_sizes`. `variants` can optionally restrict the generated configurations, for example `["original", "adaptive_pos"]`.

Use a separate `out_dir` when changing experiments whose existing results should be preserved.

## Cross-validation and preprocessing

- The script uses shuffled `KFold` with `random_state=42` by default. All variants and chunk sizes for a task use the same fold partitions.
- Under the default 10-fold protocol, approximately 90% of samples train each model and 10% form its held-out test fold.
- SNP missing values are imputed using training-fold column modes. A genotype `MinMaxScaler` with range `(-1, 1)` and a phenotype `StandardScaler` are also fitted on the training fold, then applied to both partitions.
- Each fold trains for all 150 epochs by default, with no inner validation split or early stopping. Training and test metrics are computed after the last epoch.
- Python, NumPy, and PyTorch seeds are set. A fold uses `seed + fold_number`; the same fold seed is reused across variants and chunk sizes. cuDNN deterministic mode is enabled and benchmarking is disabled.
- If the last training batch would contain one sample, one shuffled sample is omitted for that epoch to meet the model's batch-normalization requirement.

All preprocessed training and test tensors for a fold are placed on the selected device. The default requires CUDA. Set `device` to `"auto"` for automatic CUDA/CPU selection, or `"cpu"` for CPU execution.

The full-SNP shortcut contains `D × 4096` weights in its first layer, where `D` is the SNP count. Reducing chunk size does not reduce this layer's memory requirement.

## Metrics and outputs

The script reports test **Pearson correlation** and **coefficient of determination**:

$$
R^2 = 1 - \frac{\sum_i (\hat{y}_i-y_i)^2}{\sum_i(y_i-\bar{y})^2}.
$$

`R2` is computed from prediction residuals and is not the square of Pearson correlation. Metrics use standardized phenotype values; Pearson and this $R^2$ are unchanged by a common inverse standardization. `Test_Loss` is the composite training objective, rather than phenotype MSE alone.

Outputs are written relative to `CONFIG["out_dir"]`:

| Relative path | Contents |
| --- | --- |
| `<Dataset>_<Trait>/settings.json` | Task metadata, selected configurations, input paths, sample/SNP counts, and configuration signature |
| `<Dataset>_<Trait>/fold_results.csv` | Fold-level test/train metrics, losses, chunk diagnostics, parameter counts, timing, and peak allocated GPU memory |
| `<Dataset>_<Trait>/summary.csv` | Per-variant and chunk-size summaries, including fold means and sample standard deviations |
| `<Dataset>_<Trait>/chunk_importance/<variant>_C<C>_Fold<k>.csv` | Mean chunk weights over held-out samples, when adaptive weighting and weight export are enabled |
| `<Dataset>_<Trait>/checkpoints/<variant>_C<C>_Fold<k>_FinalEpoch.pth` | Final-epoch model `state_dict`, only when `keep_checkpoints=True` |
| `<Dataset>_<Trait>/Pearson_by_chunk_size.png` | Pearson comparison across chunk sizes |
| `<Dataset>_<Trait>/C<ablation_c>_Component_Ablation.png` | Component comparison at `ablation_c`, when results exist at that size |
| `all_tasks_summary.csv` | Combined summaries from the tasks selected in the current invocation |
| `All_Datasets_Pearson_by_chunk_size.png` | Combined plot for `original` and `adaptive_pos` |

Figures require Matplotlib and `no_plot=False`. Fold CSVs and task summaries are updated after each completed fold. Check `Completed_Folds` before interpreting a summary as a complete cross-validation result.

### Chunk-weight diagnostics

Adaptive variants report the mean within-sample chunk-weight standard deviation and normalized entropy over held-out samples. Exported chunk tables contain:

- `Chunk_Index_1Based`
- `SNP_Start_1Based`
- `SNP_End_1Based`
- `Mean_Weight`

SNP ranges refer to input column positions, rather than chromosome/base-pair coordinates. These weights describe the learned weighting mechanism; they do not establish causal SNP effects.

### Resume behavior

With `resume=True`, the script checks the existing fold CSV's configuration signature and skips completed `(variant, chunk size, fold)` entries. A signature mismatch raises an error.

An interrupted fold restarts from the beginning. Optional checkpoint files contain model weights only; the script does not save optimizer/scheduler states, fitted preprocessors, or per-sample predictions, and does not implement checkpoint-based inference or epoch-level training recovery.
