# -*- coding: utf-8 -*-
"""Post-hoc epistatic interaction specificity test, ORIGINAL 128D PerfVAE.

Reads the existing 10-fold `epistatic_fold*.pth` weights and source artifacts
produced by PerfVAE_Wheat2000_Full128D_Additive_Epistatic_EffectRecovery_GPU.py.
Never trains a model, never uses test phenotypes to fit/select anything.

True positives: the 3 predeclared simulated interacting SNP pairs.
Hard 1: two true interacting SNP endpoints, but not their assigned partner.
Hard 2: one true interacting SNP endpoint paired with a noncausal SNP.
Ordinary controls: two noncausal SNPs, as in the preceding experiment.

All counterfactuals use the same train-supported genotype states and held-out
individual backgrounds. This tests the specificity of the prediction function's
SNPxSNP interactions; it is not biological causal validation.

Change only CONFIG. No CLI parameters. Requires the PREVIOUS ORIGINAL SCRIPT
in the same directory for exact original architecture/inference functions.
"""
from __future__ import annotations
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG = {
    "original_script": str(SCRIPT_DIR / "PerfVAE_Wheat2000_Full128D_Additive_Epistatic_EffectRecovery_GPU.py"),
    "original_results_dir": str(SCRIPT_DIR / "Wheat2000_FullPerfVAE_Additive_Epistatic_EffectRecovery_128D"),
    "data_dir": str(SCRIPT_DIR / "data"),
    "genotype_filename": "2000gene.pkl",
    "panel": "controlled_1024",
    "scenario": "epistatic",
    "folds_to_run": list(range(1, 11)),  # Can use [1, 2] for a short pilot.
    "seed": 42,
    "n_hard_causal_background": 30, # 5 per epistatic endpoint where possible.
    "n_ordinary_background_background": 50,
    "max_pair_abs_corr_for_negatives": 0.20,  # eligible negatives only, train-fold genotypes
    "max_test_backgrounds": 200,
    "counterfactual_batch_size": 512,
    "device": "cuda",
    "allow_tf32": False,
    "check_original_positive_estimates": True,
    "original_positive_tolerance": 0.0005,
    "resume": True,
    "out_dir": str(SCRIPT_DIR / "Wheat2000_Epistatic_HardNegative_PostHoc_128D"),
}


def load_original_code(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing the prior model script: {p}; copy the provided original script next to this file")
    spec = importlib.util.spec_from_file_location("perfvae_original_effect_recovery", p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for part in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_csv(df, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g")
    os.replace(tmp, p)


def atomic_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) != 2 or not np.all(np.isfinite(scores)):
        return math.nan, math.nan
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))


def paired_scores(rows, positive_name="true_interaction"):
    pos = rows[rows.Control_Type == positive_name]
    if pos.empty:
        raise ValueError("No valid positives")
    results = {}
    score_key = "Mean_Absolute_Interaction"
    for group_name, subset in [
        ("cross_causal", rows[rows.Control_Type.isin([positive_name,"cross_causal"])]),
        ("causal_background", rows[rows.Control_Type.isin([positive_name,"causal_background"])]),
        ("ordinary", rows[rows.Control_Type.isin([positive_name,"ordinary"])]),
        ("hard_combined", rows[rows.Control_Type.isin([positive_name,"cross_causal","causal_background"])]),
        ("all_controls", rows),
    ]:
        labels = (subset.Control_Type == positive_name).astype(int).to_numpy()
        scores = subset[score_key].to_numpy(float)
        auc, ap = safe_auc(labels, scores)
        out = {f"{group_name}_AUROC": auc, f"{group_name}_AUPRC": ap,
               f"{group_name}_Nneg": int(np.sum(labels == 0))}
        negscores = scores[labels == 0]
        if len(negscores):
            out[f"{group_name}_Max_Neg_Score"] = float(np.max(negscores))
            out[f"{group_name}_Median_Neg_Score"] = float(np.median(negscores))
            out[f"{group_name}_Positive_Min_to_Negative_Max_Ratio"] = float(
                np.min(pos[score_key].to_numpy(float)) / max(float(np.max(negscores)), 1e-12))
        if len(scores) >= len(pos) and np.sum(labels == 0):
            sorted_indexes = np.argsort(-scores, kind="stable")
            out[f"{group_name}_Recall_at_{len(pos)}"] = float(
                np.sum(labels[sorted_indexes[:len(pos)]]) / len(pos))
        results.update(out)
    return results


def select_pairs(raw_train, marker_ids, truth, orig, gen_cfg, conf, fold):
    """Selection uses train-fold genotype support and fixed RNG, never predictions/test traits."""
    n = raw_train.shape[1]
    orig_ids = np.asarray(marker_ids, dtype=int)
    mapping = {int(s): i for i, s in enumerate(orig_ids)}
    true_pairs = [tuple(map(int, v)) for v in truth["pair_coef"].keys()]
    true_sets = {frozenset(pair) for pair in true_pairs}
    causal_endpoints = sorted({p for pair in true_pairs for p in pair})
    all_causal = set(truth["genotype_stats"].keys())
    if any(p not in mapping for p in causal_endpoints):
        raise ValueError("Panel does not include all interaction causal loci")
    background = np.asarray([int(v) for v in orig_ids if int(v) not in all_causal], dtype=int)
    if len(background) < 2:
        raise ValueError("No background SNPs in panel")
    rng = np.random.default_rng(int(conf["seed"]) + int(fold) * 7919 + 481)
    rng.shuffle(background)

    cache = {}
    def accepted(i, j):
        key = tuple(sorted((int(i), int(j))))
        if key in cache:
            return cache[key]
        a = mapping[key[0]]; b = mapping[key[1]]
        states = orig.possible_pair_states(raw_train, a, b, gen_cfg)
        if states is None:
            cache[key] = None
            return None
        va = raw_train[:, a].astype(float)
        vb = raw_train[:, b].astype(float)
        ok = np.isfinite(va) & np.isfinite(vb)
        rho = float(np.corrcoef(va[ok], vb[ok])[0,1]) if int(ok.sum()) > 2 else 1.0
        cache[key] = (states, abs(rho))
        return cache[key]

    chosen = []
    used_keys = set()
    skipped = []
    def add(i, j, label, enforce_ld=False):
        if int(i) == int(j):
            return False
        key = tuple(sorted((int(i), int(j))))
        if key in used_keys:
            return False
        if label != "true_interaction" and frozenset(key) in true_sets:
            raise AssertionError("A true pair was accidentally included among negative controls")
        valid = accepted(*key)
        if valid is None:
            skipped.append(dict(SNP1=key[0],SNP2=key[1],Category=label,Reason="insufficient 2x2 joint support"))
            return False
        states, rho = valid
        if enforce_ld and rho > float(conf["max_pair_abs_corr_for_negatives"]):
            return False
        chosen.append((key[0], key[1],label, states, rho))
        used_keys.add(key)
        return True

    # All real simulated pairs, independently of score.
    for i,j in true_pairs:
        if not add(i,j,"true_interaction"):
            raise ValueError(f"True pair {(i,j)} has insufficient training support")

    # Cross pairs of TRUE interaction SNPs with a WRONG true-interaction partner.
    # Six endpoints from three original true pairs yield 12 different wrong pairs.
    expected_cross=0
    for p,q in itertools.combinations(causal_endpoints,2):
        if frozenset((p,q)) not in true_sets:
            expected_cross += 1
            add(p,q,"cross_causal",enforce_ld=False)

    # One genuine epistatic endpoint + SNP with no simulated genetic effect.
    # Balance the number of such pairs across causal endpoints.
    target=int(conf["n_hard_causal_background"])
    accepted_single=0
    if target:
        pools = {k: rng.permutation(background) for k in causal_endpoints}
        offsets = {k: 0 for k in causal_endpoints}
        while accepted_single < target:
            made_progress = False
            for cause in causal_endpoints:
                arr=pools[cause]
                while offsets[cause] < len(arr):
                    b=int(arr[offsets[cause]])
                    offsets[cause] += 1
                    if add(cause,b,"causal_background",enforce_ld=True):
                        accepted_single += 1
                        made_progress = True
                        break
                if accepted_single >= target:
                    break
            if not made_progress:
                raise ValueError(f"Only {accepted_single}/{target} qualifying hard single-endpoint negatives")

    # Two background SNPs: ordinary/easier negative control.
    target_easy=int(conf["n_ordinary_background_background"])
    ordinary=0
    shuffled=rng.permutation(background)
    # Try disjoint pairs first, then diversify pairings if needed.
    for i,j in zip(shuffled[::2],shuffled[1::2]):
        if ordinary>=target_easy:
            break
        if add(int(i),int(j),"ordinary",enforce_ld=True):
            ordinary+=1
    if ordinary<target_easy:
        for _ in range(max(target_easy*50,1000)):
            i,j=rng.choice(background,size=2,replace=False)
            if add(int(i),int(j),"ordinary",enforce_ld=True):
                ordinary += 1
            if ordinary>=target_easy:
                break
    if ordinary<target_easy:
        raise ValueError(f"Only {ordinary}/{target_easy} qualifying ordinary negatives")
    if len([v for v in chosen if v[2]=="cross_causal"]) == 0:
        raise ValueError(f"No train-supported cross-causal hard negatives among {expected_cross} candidates")
    return chosen, pd.DataFrame(skipped)


def check_old_positive_estimates(df, old_path, tol):
    if not Path(old_path).exists():
        return "Previous positive CSV not found; skipped repeatability check"
    old = pd.read_csv(old_path)
    for row in df[df.Control_Type=="true_interaction"].itertuples(index=False):
        a=int(row.SNP1_Original_Index_0Based); b=int(row.SNP2_Original_Index_0Based)
        keep=old[((old.SNP1_Original_Index_0Based==a)&(old.SNP2_Original_Index_0Based==b))|
                 ((old.SNP1_Original_Index_0Based==b)&(old.SNP2_Original_Index_0Based==a))]
        if len(keep)!=1:
            raise ValueError(f"Original CSV lacks unique true pair {(a,b)}")
        earlier=keep.iloc[0]
        if abs(float(earlier.Predicted_Interaction_Effect)-float(row.Predicted_Interaction_Effect))>tol:
            raise ValueError(f"Positive pair {(a,b)} changed since original evaluation: old={earlier.Predicted_Interaction_Effect}, new={row.Predicted_Interaction_Effect}. Check checkpoint/preprocessing/source data")
    return "Exact prior positive-pair estimates reproduced within tolerance"


def analyze_fold(orig, model, tr_raw, test_backgrounds, marker_ids, truth,
                 gen_cfg, conf, fold, scenario, yscaler, transform, device):
    chosen, skipped = select_pairs(tr_raw, marker_ids, truth, orig, gen_cfg, conf, fold)
    mapping = {int(v): k for k,v in enumerate(marker_ids)}
    rows = []
    t0 = time.perf_counter()
    for idx,(i,j,typ,st,rho) in enumerate(chosen,1):
        lo_i,hi_i,lo_j,hi_j,support=st
        a=mapping[i];b=mapping[j]
        # Contrast on held-out genomic contexts: f(hi,hi)-f(hi,lo)-f(lo,hi)+f(lo,lo)
        p = orig.predicted_counterfactuals(model,test_backgrounds,transform,yscaler,
          [{a:hi_i,b:hi_j},{a:hi_i,b:lo_j},{a:lo_i,b:hi_j},{a:lo_i,b:lo_j}],device,gen_cfg)
        response=p[0]-p[1]-p[2]+p[3]
        true=orig.true_pair_delta(truth,gen_cfg,scenario,i,j,lo_i,hi_i,lo_j,hi_j)
        if typ != "true_interaction" and abs(true)>1e-9:
            raise AssertionError(f"A negative was assigned nonzero simulated interaction! {(i,j)}")
        rows.append(dict(Fold=int(fold),Control_Type=typ,Is_Known_Interaction=int(typ=="true_interaction"),
          SNP1_Original_Index_0Based=int(i),SNP2_Original_Index_0Based=int(j),
          SNP1_Panel_Index_0Based=int(a),SNP2_Panel_Index_0Based=int(b),
          SNP1_Low=float(lo_i),SNP1_High=float(hi_i),SNP2_Low=float(lo_j),SNP2_High=float(hi_j),
          Train_Min_Joint_Genotype_Count=int(min(support)),Train_Joint_Counts=str(list(map(int,support))),
          Train_Pair_Abs_Corr=float(rho),True_Interaction_Effect=float(true),
          Predicted_Interaction_Effect=float(response.mean()),Predicted_Interaction_SD=float(response.std()),
          Mean_Absolute_Interaction=float(np.mean(np.abs(response))),
          Mean_Absolute_Interaction_per_Unit_Genotype_Change=float(
              np.mean(np.abs(response))/max((hi_i-lo_i)*(hi_j-lo_j),1e-12)),
          Abs_SignedMean_Interaction=float(abs(response.mean())),
          Effect_Error=float(response.mean()-true),N_Heldout_Backgrounds=len(test_backgrounds)))
        if idx%20==0:
            print(f"    fold {fold}: scored {idx}/{len(chosen)} pairs",flush=True)
    df=pd.DataFrame(rows)
    pos=df[df.Is_Known_Interaction==1]
    reff=float(np.corrcoef(pos.True_Interaction_Effect,pos.Predicted_Interaction_Effect)[0,1]) if len(pos)>2 else math.nan
    mean_abs_true=float(pos.True_Interaction_Effect.abs().mean()); mean_abs_pred=float(pos.Predicted_Interaction_Effect.abs().mean())
    summary=dict(Fold=int(fold),N_Positive=len(pos),N_Cross_Causal=int((df.Control_Type=="cross_causal").sum()),
      N_Causal_Background=int((df.Control_Type=="causal_background").sum()),
      N_Ordinary=int((df.Control_Type=="ordinary").sum()),
      Mean_True_Abs_Effect=mean_abs_true,Mean_Predicted_Abs_Effect=mean_abs_pred,
      Effect_Abs_Magnitude_Ratio=mean_abs_pred/max(mean_abs_true,1e-12),
      Effect_Sign_Accuracy=float(np.mean(np.sign(pos.True_Interaction_Effect)==np.sign(pos.Predicted_Interaction_Effect))),
      Effect_Correlation=reff,Effect_MAE=float(np.abs(pos.Effect_Error).mean()),
      Min_Positive_Score=float(pos.Mean_Absolute_Interaction.min()),
      Median_Positive_Score=float(pos.Mean_Absolute_Interaction.median()),
      Analysis_Seconds=float(time.perf_counter()-t0))
    summary.update(paired_scores(df))
    # Robustness: divide by genotype-state displacement product, avoiding
    # artificial ranking advantages from 0->2 versus 0->1 interventions.
    df_unit=df.copy()
    df_unit["Mean_Absolute_Interaction"]=df_unit["Mean_Absolute_Interaction_per_Unit_Genotype_Change"]
    summary.update({"PerUnit_"+k:v for k,v in paired_scores(df_unit).items()})
    return df, summary, skipped


def summarize_results(out_dir):
    csvs=sorted(Path(out_dir).glob("pairs_epistatic_fold*.csv"))
    summary_csvs=sorted(Path(out_dir).glob("metrics_epistatic_fold*.json"))
    if not summary_csvs:
        return
    records=[json.loads(p.read_text(encoding="utf-8")) for p in summary_csvs]
    df=pd.DataFrame(records).sort_values("Fold")
    atomic_csv(df,Path(out_dir)/"fold_metrics.csv")
    numeric=df.select_dtypes(include=[np.number]).columns
    fields=[v for v in numeric if v!="Fold"]
    table=[]
    for name in fields:
        arr=pd.to_numeric(df[name],errors="coerce").to_numpy(float)
        finite=arr[np.isfinite(arr)]
        table.append(dict(Metric=name,Mean=float(finite.mean()) if len(finite) else math.nan,
                          Std=float(finite.std(ddof=1)) if len(finite)>1 else math.nan,
                          Valid_Folds=len(finite)))
    atomic_csv(pd.DataFrame(table),Path(out_dir)/"summary_mean_sd.csv")
    allpairs=pd.concat((pd.read_csv(p) for p in csvs),ignore_index=True)
    atomic_csv(allpairs,Path(out_dir)/"all_pairs_foldwise.csv")
    print("  Results:",Path(out_dir)/"fold_metrics.csv",flush=True)


def main():
    conf=dict(CONFIG)
    source=Path(conf["original_results_dir"])
    panel=source/conf["panel"]
    prov_path=panel/"settings_and_provenance.json"
    if not prov_path.exists():
        raise FileNotFoundError(f"Missing {prov_path}; original effect-recovery experiment must run first")
    provenance=json.loads(prov_path.read_text(encoding="utf-8"))
    train_cfg=dict(provenance["CONFIG"])
    if train_cfg["latent_dim"] != 128:
        raise ValueError("Expected original 128D PerfVAE model")
    if train_cfg["seed"] != int(conf["seed"]):
        raise ValueError("Seed differs from original run")
    if int(train_cfg["folds"]) != 10:
        raise ValueError("Expected original 10-fold CV")
    if conf["scenario"] != "epistatic":
        raise ValueError("This experiment is only for the original epistatic scenario")
    if int(conf["max_test_backgrounds"]) < 1:
        raise ValueError("Need >=1 held-out background")
    if any(int(f)<1 or int(f)>10 for f in conf["folds_to_run"]):
        raise ValueError("folds_to_run must be within 1..10")

    orig=load_original_code(conf["original_script"])
    conf_data=dict(train_cfg)
    conf_data["data_dir"]=conf["data_dir"]
    conf_data["genotype_filename"]=conf["genotype_filename"]
    X,geno_path=orig.load_wheat2000(conf_data)
    if sha256_file(geno_path)!=provenance["genotype_sha256"]:
        raise ValueError("Genotype file SHA256 differs from the original training data: DO NOT proceed")
    manifest=pd.read_csv(panel/"selected_snp_manifest.csv")
    ids=manifest.Original_SNP_Index_0Based.to_numpy(np.int64)
    if (len(ids)!=int(conf["panel"].split("_")[-1]) or
            not np.array_equal(manifest.Panel_SNP_Index_0Based.to_numpy(int),np.arange(len(ids)))):
        raise ValueError("Invalid panel marker manifest")
    if hashlib.sha256(ids.tobytes()).hexdigest()!=provenance["panel_marker_sha256"]:
        raise ValueError("Marker manifest checksum mismatch")
    assignments=pd.read_csv(source/"cv_fold_assignments.csv")
    if not np.array_equal(assignments.Sample_Row_Index_0Based.to_numpy(int),np.arange(len(X))):
        raise ValueError("CV sample row order has changed")
    cv=list(KFold(n_splits=10,shuffle=True,random_state=int(conf["seed"])).split(X))
    for fold,(train,test) in enumerate(cv,1):
        if not np.all(assignments.Heldout_Fold_1Based.to_numpy()[test]==fold):
            raise ValueError(f"CV fold {fold} does not agree with original saved assignment")
    causal={k:pd.read_csv(source/(k+".csv")) for k in ("additive_snp","epistatic_pairs")}
    truth=orig.make_truth_spec(X,causal,train_cfg)
    sim=pd.read_csv(source/"controlled_simulated_phenotypes.csv")
    if (len(sim)!=len(X) or not np.array_equal(sim.Sample_Row_Index_0Based.to_numpy(int),np.arange(len(X)))):
        raise ValueError("Simulated phenotype source row order mismatch")
    y=sim["Simulated_epistatic"].to_numpy(float)

    device=torch.device(conf["device"])
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Change CONFIG['device'] to 'cpu' for testing")
    torch.backends.cuda.matmul.allow_tf32=bool(conf["allow_tf32"])
    torch.backends.cudnn.allow_tf32=bool(conf["allow_tf32"])
    orig.seed_everything(conf["seed"])
    X_panel=np.asarray(X[:,ids],dtype=np.float32)
    out=Path(conf["out_dir"])
    out.mkdir(parents=True,exist_ok=True)
    signature={"original_config_id":provenance["Config_ID"],"genotype_sha256":provenance["genotype_sha256"],
       "panel_marker_sha256":provenance["panel_marker_sha256"],
       "analysis_options":{k:v for k,v in conf.items() if k not in ("folds_to_run","resume","out_dir","data_dir","original_results_dir","original_script","genotype_filename")}}
    sigpath=out/"analysis_provenance.json"
    if sigpath.is_file():
        existing=json.loads(sigpath.read_text(encoding="utf-8"))
        if existing!=signature:
            raise ValueError("Previously saved analysis has different settings! Use a new out_dir")
    else:
        atomic_json(signature,sigpath)
    print("="*76,flush=True)
    print("Wheat2000 | 128D FULL PerfVAE | post-hoc hard-negative epistasis validation",flush=True)
    print("NO RETRAINING | panel",conf["panel"],"| folds",conf["folds_to_run"],flush=True)
    print("Original results:",source,flush=True)
    print("New results:",out,flush=True)
    print("="*76,flush=True)
    for fold,(tr,te) in enumerate(cv,1):
        if fold not in conf["folds_to_run"]:
            continue
        detailed=out/f"pairs_epistatic_fold{fold}.csv"
        summarized=out/f"metrics_epistatic_fold{fold}.json"
        if bool(conf["resume"]) and detailed.exists() and summarized.exists():
            print(f"Fold {fold}: completed, skip",flush=True)
            continue
        checkpoint=panel/"checkpoints"/f"epistatic_fold{fold}.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Fold {fold} checkpoint missing: {checkpoint}\n"
                    "Original run must have save_checkpoints=True. Raw fold CSV alone cannot reconstruct NN predictions.")
        ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if ck.get("Config_ID")!=provenance["Config_ID"]:
            raise ValueError(f"Checkpoint signature mismatch: {checkpoint}")
        model=orig.make_nn("full_perfvae",len(ids),train_cfg).to(device)
        model.load_state_dict(ck["state_dict"],strict=True)
        model.eval()
        trraw=X_panel[tr]
        _, _, transform=orig.fitted_x_preprocess(trraw,X_panel[te])
        yscaler=orig.raw_y_scaler(y[tr])
        contexts=np.asarray(X_panel[te][:int(conf["max_test_backgrounds"])],dtype=np.float32)
        inference_config=dict(train_cfg)
        inference_config["counterfactual_batch_size"]=int(conf["counterfactual_batch_size"])
        print(f"Fold {fold}: loaded saved weights; testing {len(contexts)} held-out contexts",flush=True)
        foldtab, metrics, skipped=analyze_fold(orig,model,trraw,contexts,ids,truth,
                    inference_config,conf,fold,"epistatic",yscaler,transform,device)
        if conf["check_original_positive_estimates"]:
            msg=check_old_positive_estimates(foldtab,
                panel/f"interaction_effects_epistatic_fold{fold}.csv",float(conf["original_positive_tolerance"]))
            print("  ",msg,flush=True)
        atomic_csv(foldtab,detailed)
        atomic_json(metrics,summarized)
        if len(skipped):
            atomic_csv(skipped,out/f"skipped_unsupported_cross_pairs_fold{fold}.csv")
        print(f"  Fold {fold} AUROC: ordinary={metrics['ordinary_AUROC']:.3f} "
              f"cross={metrics['cross_causal_AUROC']:.3f} "
              f"single={metrics['causal_background_AUROC']:.3f} "
              f"ALL={metrics['all_controls_AUROC']:.3f}; "
              f"AUPRC(all)={metrics['all_controls_AUPRC']:.3f}",flush=True)
        del model
        if device.type=="cuda":
            torch.cuda.empty_cache()
        summarize_results(out)
    summarize_results(out)
    print("Finished. Post-hoc only: NO ORIGINAL NETWORK OR TRAINING WAS MODIFIED.",flush=True)

if __name__=="__main__":
    main()
