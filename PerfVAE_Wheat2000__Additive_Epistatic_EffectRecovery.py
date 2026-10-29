# -*- coding: utf-8 -*-
"""Wheat2000 | 128-D Original PerfVAE | Additive + Epistatic Effect Recovery.

No rrBLUP, no ablations: train ONLY original 128-D Full PerfVAE.
Simulates additive/epistatic phenotype using same fixed genotype, causal seed,
marker panel and CV split as previously provided Wheat2000 Oracle Controlled script.
Evaluates counterfactual main effects and pairwise double differences on HELD-OUT
samples, with train-supported genotype states, and known additive/interaction truth.
The interventions are prediction-function perturbations, not causal biological proofs.

No command-line arguments. Edit CONFIG only. See paired README.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
try:
    from performer_pytorch import Performer
except ImportError:
    Performer=None

SCRIPT_DIR=Path(__file__).resolve().parent
CONFIG={
    "data_dir":str(SCRIPT_DIR/"data"),
    "genotype_filename":"2000gene.pkl",
    "out_dir":str(SCRIPT_DIR/"Wheat2000_FullPerfVAE_Additive_Epistatic_EffectRecovery_128D"),
    # Same synthetic genetic architecture as earlier CONTROLLED experiment.
    "controlled_marker_counts":[1024],   # Optional [1024,2048]; same phenotype and causal SNPs
    "controlled_additive_snps":8,
    "controlled_epistatic_pairs":3,
    "controlled_causal_seed":2027,
    "scenarios":{"additive":0.0,"mixed":0.5,"epistatic":1.0},
    "scenarios_to_run":["additive","epistatic"],  # "mixed" optional, trains another model
    "heritability":0.70,
    "min_marker_call_rate":0.98,
    "min_marker_std":0.30,
    "max_pair_abs_corr":0.10,
    "min_pair_column_distance":256,
    "max_additive_interaction_corr":0.08,
    # Preserved Wheat2000 original full-model training settings.
    "folds":10,
    "folds_to_run":[1,2], # Change to list(range(1,11)) after pilot; same 10-fold split
    "seed":42,
    "epochs":150,
    "batch_size":64,
    "chunk_size":256,
    "latent_dim":128,
    "performer_dim":256,
    "dropout":0.4,
    "lr":1e-5,
    "max_lr":5e-4,
    "weight_decay":1e-5,
    "device":"cuda", # "cpu" for smoke test only
    "fused_adamw":True,
    "tf32":False,
    "epoch_log_interval":0,
    # Main-effect negative controls: SNPs with NO simulated additive/epistatic role.
    "n_additive_negative_snps":40,
    # Interaction negatives: both markers have no simulated causal role.
    "n_epistatic_negative_pairs":50,
    "min_genotype_count":10, # each selected genotype state among outer TRAIN genotypes
    "min_pair_joint_count":10, # each of four genotype combinations in outer TRAIN
    "max_test_backgrounds":200, # independent held-out genotypes only
    "counterfactual_batch_size":256,
    "save_checkpoints":True,
    "make_plots":True, # requires matplotlib; skipped safely if unavailable
    "resume":True,
}

class GeneticAwareLoss(nn.Module):
    """Identical effective weights and reduction to the submitted code."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x_recon, x, y_pred, y, mu, logvar, aux_logits=None, aux_targets=None):
        # FP32 loss math protects MSE and KL even when BF16/FP16 autocast is used.
        y_pred, x_recon = y_pred.float(), x_recon.float()
        mu, logvar = mu.float(), logvar.float()
        pred = self.mse(y_pred, y.float())
        recon = self.mse(x_recon, x.float())
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

def seed_everything(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def genotype_path(c):
    root=Path(c['data_dir'])
    for fp in (root/c['genotype_filename'],root/'wheat2000'/c['genotype_filename'],
               root/'Wheat2000'/c['genotype_filename']):
        if fp.is_file():return fp
    raise FileNotFoundError(f"Missing Wheat2000 genotype {c['genotype_filename']} under {root}")

def load_wheat2000(c):
    fp=genotype_path(c)
    obj=pd.read_pickle(fp)
    X=np.asarray(obj.to_numpy() if hasattr(obj,'to_numpy') else obj,dtype=np.float32)
    if X.ndim!=2 or X.shape[0]<c['folds']*3 or X.shape[1]<50:
        raise ValueError(f"Expected [samples,SNP] Wheat2000 genotype matrix; got {X.shape}")
    if np.isinf(X).any(): raise ValueError("Genotype contains +/-Inf")
    return np.ascontiguousarray(X),fp

def standardize_array(x):
    x=np.asarray(x,dtype=np.float64)
    x=x-x.mean()
    s=np.sqrt(np.mean(x*x))
    if s<1e-10: raise ValueError("Simulated genetic component has near-zero variance")
    return x/s

def simulate_architectures(X,c):
    """One shared causal architecture; 3 phenotype scenarios.

    SNP selection uses ONLY real genotypes and fixed RNG, not observed trait
    labels. The synthetic phenotype is defined globally before CV. This is
    phenotype DEFINITION, not genotype preprocessing for model fitting.
    No full-matrix global imputation or SNP scaling is used in fitting.
    """
    rng=np.random.default_rng(int(c['causal_seed']))
    n,p=X.shape
    good=np.isfinite(X)
    valid_rate=good.mean(axis=0)
    # Only eligible SNPs need accurate mean/variance; normal operation p~33k.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        gstd=np.nanstd(X,axis=0)
    eligible=np.flatnonzero((valid_rate>=float(c['min_marker_call_rate'])) &
                            (gstd>=float(c['min_marker_std'])))
    a=int(c['n_additive_snps']); e=int(c['n_epistatic_pairs'])
    if len(eligible)<a+2*e: raise ValueError(
        f"Only {len(eligible)} eligible polymorphic SNPs; need {a+2*e}. "
        "Adjust min_marker_std/call_rate for this file, or inspect genotype coding.")
    # Gaussian coefficient magnitudes, with signs and nontrivial effects.
    # Same causal main-effect panel is used in all 3 scenarios.
    add_ids=np.sort(rng.choice(eligible,size=a,replace=False))
    add_beta=rng.choice([-1.,1.],size=a)*rng.uniform(0.7,1.3,size=a)
    def select_z(ids):
        block=np.asarray(X[:,ids],dtype=np.float64)
        means=np.nanmean(block,axis=0)
        block=np.where(np.isfinite(block),block,means)
        sd=np.std(block,axis=0)
        if np.any(sd<1e-8): raise ValueError("Unusable causal SNP after filtering")
        return (block-block.mean(axis=0))/sd
    za=select_z(add_ids)
    add_effect=standardize_array(za @ add_beta)
    # Prefer distant-in-input-order and empirically weakly correlated SNP pairs.
    # Marker index distance is NOT physical/genetic distance.
    pair_ids=[]; used=set(add_ids.tolist())
    candidates=rng.permutation(eligible)
    cursor=0
    for first in candidates:
        first=int(first)
        if first in used:continue
        z1=select_z([first])[:,0]
        # Keep selecting until we find low-LD partners; input-order separation
        # is only a heuristic when chromosome metadata is absent.
        for partner in rng.choice(eligible,size=min(150,len(eligible)),replace=False):
            partner=int(partner)
            if partner==first or partner in used:continue
            if abs(first-partner)<int(c['min_pair_column_distance']):continue
            z2=select_z([partner])[:,0]
            rho=abs(float(np.mean(z1*z2)))
            if rho<=float(c['max_pair_abs_corr']):
                pair_ids.append((first,partner,rho));used.add(first);used.add(partner)
                break
        if len(pair_ids)>=e:break
    if len(pair_ids)<e: raise ValueError(
        f"Could find only {len(pair_ids)} low-correlation SNP pairs. "
        "Check genotype variability or relax the pair criteria.")
    gammas=rng.choice([-1.,1.],size=e)*rng.uniform(.7,1.3,size=e)
    interaction_mat=np.column_stack([
        select_z([i])[:,0]*select_z([j])[:,0] for i,j,_ in pair_ids])
    interaction=standardize_array(interaction_mat @ gammas)
    ai_corr=float(np.mean(add_effect*interaction))
    # A and I should be close to orthogonal for eta to represent distinct
    # nominal variance fractions. No data-dependent orthogonalization of I
    # (which would alter the defined SNPxSNP interaction function) is used.
    if abs(ai_corr)>float(c['max_additive_interaction_corr']):
        print(f"WARNING: corr(A,I)={ai_corr:.4f} exceeds "
              f"{c['max_additive_interaction_corr']}; eta denotes NOMINAL "
              "weights, not exactly identified variance shares.",flush=True)
    eps=standardize_array(rng.normal(size=n))
    Y={};manifest=[]
    h2=float(c['heritability'])
    for name,eta0 in c['scenarios'].items():
        eta=float(eta0)
        genetic=standardize_array(math.sqrt(1-eta)*add_effect+math.sqrt(eta)*interaction)
        genetic=math.sqrt(h2)*genetic
        noise=math.sqrt(1-h2)*eps
        phen=genetic+noise
        realized=float(np.var(genetic)/np.var(phen))
        Y[name]=phen.astype(np.float32).reshape(-1,1)
        manifest.append(dict(Scenario=name,Nominal_Epistatic_Fraction=eta,
                             Heritability_Target=h2,Heritability_Realized=realized,
                             Additive_Interaction_Correlation=ai_corr,
                             Phenotype_Mean=float(phen.mean()),Phenotype_SD=float(phen.std()),
                             Genetic_Noise_Correlation=float(np.corrcoef(genetic,noise)[0,1])))
    genetic_details={
      'additive_snp':pd.DataFrame({'SNP_Index_0Based':add_ids,'SNP_Index_1Based':add_ids+1,
                                   'Additive_Effect_Coefficient':add_beta}),
      'epistatic_pairs':pd.DataFrame({'SNP1_Index_0Based':[a for a,b,r in pair_ids],
                                    'SNP2_Index_0Based':[b for a,b,r in pair_ids],
                                    'SNP1_Index_1Based':[a+1 for a,b,r in pair_ids],
                                    'SNP2_Index_1Based':[b+1 for a,b,r in pair_ids],
                                    'Pair_Correlation':[r for a,b,r in pair_ids],
                                    'Epistatic_Effect_Coefficient':gammas}),
    }
    return Y,pd.DataFrame(manifest),genetic_details

def scale_phenotype(ytr_raw,yte_raw):
    sy=StandardScaler().fit(ytr_raw)
    return np.ascontiguousarray(sy.transform(ytr_raw),dtype=np.float32),\
           np.ascontiguousarray(sy.transform(yte_raw),dtype=np.float32)

def metrics_from_tensor(ytrue,ypred):
    ytrue=ytrue.float().reshape(-1)
    ypred=ypred.float().reshape(-1)
    tc=ytrue-ytrue.mean();pc=ypred-ypred.mean()
    den=torch.sqrt(torch.sum(tc**2)*torch.sum(pc**2))
    r=torch.sum(tc*pc)/den.clamp_min(1e-12)
    r2=1-torch.sum((ytrue-ypred)**2)/torch.sum(tc**2).clamp_min(1e-12)
    return float(r.item()),float(r2.item())

def minibatches(x,y,batch,shuffle=True):
    n=x.shape[0]
    indices=torch.randperm(n,device=x.device) if shuffle else torch.arange(n,device=x.device)
    # Original Wheat benchmark drops any singleton BN batch on shuffled training.
    if shuffle and len(indices)%batch==1:indices=indices[:-1]
    for st in range(0,len(indices),batch):
        select=indices[st:st+batch]
        yield x[select],y[select]

def predict_nn(model,x,batch):
    model.eval();pred=[]
    for st in range(0,len(x),batch):
        yp=model(x[st:st+batch])[1]
        pred.append(yp.flatten())
    return torch.cat(pred)

def run_nn(kind,xtr,ytr,xte,yte,c,fold_seed,device):
    if Performer is None:
        raise ImportError('Run: python -m pip install performer-pytorch')
    seed_everything(fold_seed)
    model=make_nn(kind,xtr.shape[1],c).to(device)
    params=sum(w.numel() for w in model.parameters())
    trainable=sum(w.numel() for w in model.parameters() if w.requires_grad)
    optimizer_kws=dict(lr=c['lr'],weight_decay=c['weight_decay'])
    if device.type=='cuda' and c['fused_adamw']:
        try:opt=torch.optim.AdamW(model.parameters(),fused=True,**optimizer_kws)
        except (RuntimeError,ValueError,TypeError):
            opt=torch.optim.AdamW(model.parameters(),foreach=False,**optimizer_kws)
    else:opt=torch.optim.AdamW(model.parameters(),foreach=False,**optimizer_kws)
    steps=xtr.shape[0]//c['batch_size']+int(xtr.shape[0]%c['batch_size']>1)
    if steps<1:raise ValueError('BatchNorm requires >1 training sample per step')
    schedule=torch.optim.lr_scheduler.OneCycleLR(
        opt,max_lr=c['max_lr'],steps_per_epoch=steps,epochs=c['epochs'])
    loss_fn=GeneticAwareLoss()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    t0=time.perf_counter()
    for ep in range(1,c['epochs']+1):
        model.train()
        for bx,by in minibatches(xtr,ytr,c['batch_size']):
            opt.zero_grad(set_to_none=True)
            xr,yp,mu,lv,_=model(bx)
            loss=loss_fn(xr,bx,yp,by,mu,lv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
            opt.step();schedule.step()
        if c['epoch_log_interval'] and (ep==1 or ep%c['epoch_log_interval']==0 or ep==c['epochs']):
            print(f'    {kind} epoch {ep}/{c["epochs"]}',flush=True)
    if device.type=='cuda':torch.cuda.synchronize(device)
    fit_s=time.perf_counter()-t0
    train_pred=predict_nn(model,xtr,c['batch_size'])
    test_pred=predict_nn(model,xte,c['batch_size'])
    tr_r,tr_r2=metrics_from_tensor(ytr.flatten(),train_pred)
    te_r,te_r2=metrics_from_tensor(yte.flatten(),test_pred)
    peak=(torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else np.nan)
    return dict(model=model,optimizer=opt,scheduler=schedule,params=params,trainable=trainable,
                train_r=tr_r,train_r2=tr_r2,test_r=te_r,test_r2=te_r2,
                time_s=fit_s,peak_mb=peak)

def normalized_genotypes(X,indices):
    ids=np.asarray(indices,dtype=np.int64)
    block=np.asarray(X[:,ids],dtype=np.float64)
    m=np.nanmean(block,axis=0)
    block=np.where(np.isfinite(block),block,m)
    sd=block.std(axis=0)
    if (sd<1e-10).any():raise ValueError('Near-constant causal genotype')
    return (block-block.mean(axis=0))/sd

def genetic_truth_from_saved_ids(X,causal,c):
    """Reconstruct exact original Stage1 noiseless genetic means from saved truth.

    The stage1 generator globally mean-imputed and centered each causal SNP,
    standardized A and I separately, then standardized their weighted sum.
    This uses ALL genotype rows ONLY to reconstruct the SYNTHETIC TRUE
    generative function; it is never used for fitted test-set preprocessing.
    """
    a=causal['additive_snp']
    e=causal['epistatic_pairs']
    A=normalized_genotypes(X,a['SNP_Index_0Based'].astype(int).to_numpy())@a['Additive_Effect_Coefficient'].to_numpy(float)
    A=standardize_array(A)
    ix=e['SNP1_Index_0Based'].astype(int).to_numpy()
    jx=e['SNP2_Index_0Based'].astype(int).to_numpy()
    zi=normalized_genotypes(X,ix)
    zj=normalized_genotypes(X,jx)
    I=standardize_array((zi*zj)@e['Epistatic_Effect_Coefficient'].to_numpy(float))
    g={}
    for s,eta in c['scenarios'].items():
        full=standardize_array(math.sqrt(1-float(eta))*A+math.sqrt(float(eta))*I)
        g[s]=(math.sqrt(float(c['heritability']))*full).astype(np.float64)
    return g

def controlled_simulation(X,c):
    """True phenotype defined once for ALL nested marker panels."""
    tmp=dict(c)
    tmp['causal_seed']=int(c['controlled_causal_seed'])
    tmp['n_additive_snps']=int(c['controlled_additive_snps'])
    tmp['n_epistatic_pairs']=int(c['controlled_epistatic_pairs'])
    ph,report,causal=simulate_architectures(X,tmp)
    Y={s:a[:,0].astype(np.float64) for s,a in ph.items()}
    G=genetic_truth_from_saved_ids(X,causal,tmp)
    return Y,G,causal,report

def nested_marker_panels(X,c,causal):
    """Causal SNPs included in ALL panels; remaining markers are nested background.

    Critically: selection is fixed *before model outcomes* and does not use
    simulated phenotype or test performance. Index order remains original.
    """
    a=causal['additive_snp']['SNP_Index_0Based'].astype(int).to_numpy()
    p=causal['epistatic_pairs']
    ids=np.unique(np.concatenate([a,p['SNP1_Index_0Based'].astype(int).to_numpy(),
                                   p['SNP2_Index_0Based'].astype(int).to_numpy()]))
    available=np.setdiff1d(np.arange(X.shape[1],dtype=np.int64),ids,assume_unique=True)
    rng=np.random.default_rng(int(c['controlled_causal_seed'])+9919)
    rng.shuffle(available)
    panels={}
    for size in sorted(int(s) for s in c['controlled_marker_counts']):
        if size>X.shape[1]:raise ValueError(f'{size} SNPs requested, only {X.shape[1]} in source')
        markers=np.sort(np.concatenate([ids,available[:size-len(ids)]]))
        assert len(markers)==size and np.isin(ids,markers).all()
        panels[size]=markers
    for small,big in zip(list(panels)[:-1],list(panels)[1:]):
        if not np.isin(panels[small],panels[big]).all():
            raise AssertionError('Panels not nested, simulation design inconsistent')
    return panels


# Original model body above is source-copied from the 128-D, fixed-150-epoch
# Wheat2000 experimental script. No modified model or extra loss term.

def make_nn(kind,D,c):
    if kind != 'full_perfvae':
        raise ValueError('This experiment deliberately trains ONLY the original Full PerfVAE')
    return EnhancedGeneticModel(snp_dim=D,chunk_size=c['chunk_size'],
          latent_dim=c['latent_dim'],performer_dim=c['performer_dim'],dropout=c['dropout'],
          adaptive_weighting=False,learnable_position=False)


def atomic_csv(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    df.to_csv(temp,index=False,float_format='%.12g');os.replace(temp,path)


def atomic_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def make_signature(c,genodigest,marker_ids):
    # Operational flags excluded: adding folds later does NOT invalidate 1-2.
    omit={'out_dir','data_dir','folds_to_run','resume','save_checkpoints','epoch_log_interval',
          'counterfactual_batch_size','scenarios_to_run','controlled_marker_counts','make_plots'}
    d={'CONFIG':{k:v for k,v in c.items() if k not in omit},'genotype_sha256':genodigest,
       'panel_marker_sha256':hashlib.sha256(np.asarray(marker_ids,dtype=np.int64).tobytes()).hexdigest(),
       'version':'effect-recovery-v1'}
    return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),d


def fitted_x_preprocess(Xtr,Xte):
    # Exactly the prior training-fold-only most_frequent + minmax protocol.
    imp=None
    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        try:imp=SimpleImputer(strategy='most_frequent',keep_empty_features=True)
        except TypeError:imp=SimpleImputer(strategy='most_frequent')
        arr=imp.fit_transform(Xtr)
        if arr.shape[1]!=Xtr.shape[1]:raise ValueError('Imputer dropped all-missing SNP')
    else:arr=Xtr
    scaler=MinMaxScaler(feature_range=(-1,1)).fit(arr)
    def transform(new):
        q=imp.transform(new) if imp is not None else new
        return np.ascontiguousarray(scaler.transform(q),dtype=np.float32)
    return transform(Xtr),transform(Xte),transform


def raw_y_scaler(ytrain):
    return StandardScaler().fit(np.asarray(ytrain).reshape(-1,1))


def make_truth_spec(X,causal,c):
    # Recover *exact* simulator normalization, with full-sample statistics used
    # ONLY to define synthetic truth. Training preprocessing remains train-only.
    ai=causal['additive_snp']['SNP_Index_0Based'].to_numpy(int)
    ij=causal['epistatic_pairs']
    p1=ij['SNP1_Index_0Based'].to_numpy(int)
    p2=ij['SNP2_Index_0Based'].to_numpy(int)
    all_ids=np.unique(np.r_[ai,p1,p2])
    raw=np.asarray(X[:,all_ids],dtype=np.float64)
    means=np.nanmean(raw,axis=0)
    clean=np.where(np.isfinite(raw),raw,means)
    sd=clean.std(axis=0)
    if (sd<=1e-9).any():raise ValueError('Constant causal SNP')
    m={int(snp):(float(means[k]),float(sd[k])) for k,snp in enumerate(all_ids)}
    Z={int(snp):(clean[:,k]-means[k])/sd[k] for k,snp in enumerate(all_ids)}
    b=causal['additive_snp']['Additive_Effect_Coefficient'].to_numpy(float)
    gam=ij['Epistatic_Effect_Coefficient'].to_numpy(float)
    Ar=np.column_stack([Z[int(k)] for k in ai])@b
    Ir=np.column_stack([Z[int(a)]*Z[int(d)] for a,d in zip(p1,p2)])@gam
    A=(Ar-Ar.mean())/Ar.std()
    I=(Ir-Ir.mean())/Ir.std()
    result=dict(A_std=A,I_std=I,genotype_stats=m,add_coef=dict(zip(ai,b)),
                pair_coef={(int(a),int(d)):float(g) for a,d,g in zip(p1,p2,gam)},
                A_scale=float(Ar.std()),I_scale=float(Ir.std()),weighted_sd={})
    for name,eta in c['scenarios'].items():
        mixed=math.sqrt(1-float(eta))*A+math.sqrt(float(eta))*I
        result['weighted_sd'][name]=float(mixed.std())
    return result


def true_add_delta(info,c,scenario,snp,lo,hi):
    eta=float(c['scenarios'][scenario]); beta=info['add_coef'].get(int(snp),0.0)
    if beta==0 or eta>=1:return 0.0
    mean,sd=info['genotype_stats'][int(snp)]
    return (math.sqrt(float(c['heritability']))*math.sqrt(1-eta)*beta*
            float(hi-lo)/(sd*info['A_scale']*info['weighted_sd'][scenario]))


def true_pair_delta(info,c,scenario,i,j,lo_i,hi_i,lo_j,hi_j):
    eta=float(c['scenarios'][scenario])
    if eta<=0:return 0.0
    gamma=info['pair_coef'].get((int(i),int(j)), info['pair_coef'].get((int(j),int(i)),0.0))
    if gamma==0:return 0.0
    _,si=info['genotype_stats'][int(i)];_,sj=info['genotype_stats'][int(j)]
    return (math.sqrt(float(c['heritability']))*math.sqrt(eta)*gamma*
            float(hi_i-lo_i)*float(hi_j-lo_j)/
            (si*sj*info['I_scale']*info['weighted_sd'][scenario]))


def possible_states(train_column,c):
    v=np.asarray(train_column).ravel()
    v=v[np.isfinite(v)]
    val,cnt=np.unique(v,return_counts=True)
    keep=cnt>=int(c['min_genotype_count'])
    pairs=sorted(zip(val[keep],cnt[keep]),key=lambda q:(-q[1],q[0]))
    if len(pairs)<2:return None
    # Choose common allele classes, not extreme rare SNP states.
    lo,hi=sorted([float(pairs[0][0]),float(pairs[1][0])])
    return lo,hi


def possible_pair_states(Xtr,i,j,c):
    xi=np.asarray(Xtr[:,i]);xj=np.asarray(Xtr[:,j])
    first=np.unique(xi[np.isfinite(xi)]);second=np.unique(xj[np.isfinite(xj)])
    options=[]
    for ai in range(len(first)):
        for bi in range(ai+1,len(first)):
            xlo,xhi=float(first[ai]),float(first[bi])
            if (np.sum(xi==xlo)<c['min_genotype_count'] or np.sum(xi==xhi)<c['min_genotype_count']):continue
            for aj in range(len(second)):
                for bj in range(aj+1,len(second)):
                    ylo,yhi=float(second[aj]),float(second[bj])
                    if (np.sum(xj==ylo)<c['min_genotype_count'] or np.sum(xj==yhi)<c['min_genotype_count']):continue
                    counts=[int(np.sum((xi==a)&(xj==b))) for a in (xlo,xhi) for b in (ylo,yhi)]
                    # Pick the most supported 2x2 genotype square; no test labels used.
                    if min(counts)>=int(c['min_pair_joint_count']):
                        options.append((min(counts),sum(counts),xlo,xhi,ylo,yhi,counts))
    if not options:return None
    best=max(options,key=lambda v:(v[0],v[1]))
    return best[2],best[3],best[4],best[5],best[6]


@torch.inference_mode()
def predicted_counterfactuals(model, raw_contexts, transform, yscaler, edits, device,c):
    # edits: iterable of {panel_snp_index: new raw genotype code}.
    B=len(raw_contexts);buf=[]
    for assignments in edits:
        tmp=np.array(raw_contexts,copy=True)
        for col,value in assignments.items():tmp[:,int(col)]=float(value)
        buf.append(transform(tmp))
    xx=np.concatenate(buf,axis=0)
    preds=[]
    for start in range(0,len(xx),int(c['counterfactual_batch_size'])):
        batch=torch.from_numpy(xx[start:start+c['counterfactual_batch_size']]).to(device)
        preds.append(predict_nn(model,batch,int(c['batch_size'])).detach().cpu().numpy())
    ystd=np.concatenate(preds).reshape(len(edits),B)
    return ystd*float(yscaler.scale_[0])+float(yscaler.mean_[0])


def assess_main_effects(model,raw_train,raw_contexts,ids,truth,c,scenario,
                        marker_map,transform,yscaler,device,fold,panel,seed):
    rng=np.random.default_rng(seed+777)
    pos=truth['add_coef']
    excluded=set(truth['genotype_stats'])
    controls=[int(p) for p in ids if int(p) not in excluded]
    rng.shuffle(controls)
    chosen=[]
    for original in sorted(pos):
        col=marker_map[int(original)]
        st=possible_states(raw_train[:,col],c)
        if st is None:raise ValueError(f'Causal additive SNP {original} lacks 2 common genotype states')
        chosen.append((int(original),True,st))
    nneg=0
    for original in controls:
        if nneg>=c['n_additive_negative_snps']:break
        st=possible_states(raw_train[:,marker_map[original]],c)
        if st is not None:
            chosen.append((original,False,st));nneg+=1
    if nneg<c['n_additive_negative_snps']:
        raise ValueError(f'Only {nneg} qualifying additive negative controls')
    rows=[]
    for original,positive,(lo,hi) in chosen:
        col=marker_map[original]
        p=predicted_counterfactuals(model,raw_contexts,transform,yscaler,
                                  [{col:lo},{col:hi}],device,c)
        changes=p[1]-p[0]
        actual=true_add_delta(truth,c,scenario,original,lo,hi)
        rows.append(dict(Panel=panel,Scenario=scenario,Fold=fold,
          Original_SNP_Index_0Based=original,Panel_SNP_Index_0Based=col,
          Is_Known_Additive=int(positive),Genotype_Low=lo,Genotype_High=hi,
          True_Main_Effect=actual,Predicted_Main_Effect=float(changes.mean()),
          Predicted_Effect_SD=float(changes.std()),
          Abs_Predicted_Effect=float(abs(changes.mean())),
          Effect_Error=float(changes.mean()-actual),
          N_Test_Backgrounds=len(raw_contexts)))
    return pd.DataFrame(rows)


def assess_pair_effects(model,raw_train,raw_contexts,ids,truth,c,scenario,
                        marker_map,transform,yscaler,device,fold,panel,seed):
    rng=np.random.default_rng(seed+888)
    positives=list(truth['pair_coef'])
    all_causal=set(truth['genotype_stats'])
    controls=[int(i) for i in ids if int(i) not in all_causal]
    rng.shuffle(controls)
    candidates=[(int(a),int(b),False) for a,b in zip(controls[::2],controls[1::2])]
    chosen=[]
    for i,j in positives:
        sel=possible_pair_states(raw_train,marker_map[i],marker_map[j],c)
        if sel is None:raise ValueError(f'Causal interacting pair {(i,j)} has no supported 2x2 genotype square in train fold')
        chosen.append((i,j,True,sel))
    nneg=0
    for i,j,_ in candidates:
        if nneg>=c['n_epistatic_negative_pairs']:break
        sel=possible_pair_states(raw_train,marker_map[i],marker_map[j],c)
        if sel is not None:
            chosen.append((i,j,False,sel));nneg+=1
    if nneg<c['n_epistatic_negative_pairs']:
        raise ValueError(f'Only {nneg} qualifying epistatic negatives; lower n_epistatic_negative_pairs')
    rows=[]
    for i,j,ispositive,(lo_i,hi_i,lo_j,hi_j,support) in chosen:
        a=marker_map[i];b=marker_map[j]
        # Cross difference f(++ ) - f(+-) - f(-+) + f(--).
        pred=predicted_counterfactuals(model,raw_contexts,transform,yscaler,
           [{a:hi_i,b:hi_j},{a:hi_i,b:lo_j},
            {a:lo_i,b:hi_j},{a:lo_i,b:lo_j}],device,c)
        deltas=pred[0]-pred[1]-pred[2]+pred[3]
        true=true_pair_delta(truth,c,scenario,i,j,lo_i,hi_i,lo_j,hi_j)
        rows.append(dict(Panel=panel,Scenario=scenario,Fold=fold,
          SNP1_Original_Index_0Based=i,SNP2_Original_Index_0Based=j,
          SNP1_Panel_Index_0Based=a,SNP2_Panel_Index_0Based=b,
          Is_Known_Interaction=int(ispositive),Genotype1_Low=lo_i,Genotype1_High=hi_i,
          Genotype2_Low=lo_j,Genotype2_High=hi_j,
          Train_Joint_Genotype_Min_Count=min(support),
          True_Interaction_Effect=true,
          Predicted_Interaction_Effect=float(deltas.mean()),
          Predicted_Interaction_SD=float(deltas.std()),
          Mean_Absolute_Interaction=float(np.mean(np.abs(deltas))),
          Abs_SignedMean_Interaction=float(abs(deltas.mean())),
          Effect_Error=float(deltas.mean()-true),
          N_Test_Backgrounds=len(raw_contexts)))
    return pd.DataFrame(rows)


def nan_r(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    if len(a)<2 or np.std(a)<1e-10 or np.std(b)<1e-10:return float('nan')
    return float(np.corrcoef(a,b)[0,1])


def effect_summary(main,pairs,c,fold,scenario,panel,test_r,test_r2,train_r,sec,params,configid):
    res=dict(Config_ID=configid,Panel=panel,Scenario=scenario,Fold=fold,
             Train_Pearson=train_r,Test_Pearson=test_r,Test_R2=test_r2,
             Training_Time_s=sec,Total_Params_M=params/1e6,
             Additive_N=0,Epistatic_N=0)
    if main is not None:
        caus=main[main.Is_Known_Additive==1];y=main.Is_Known_Additive.to_numpy(int)
        score=main.Abs_Predicted_Effect.to_numpy(float)
        res.update(Additive_N=len(caus),Additive_Effect_Corr=nan_r(caus.True_Main_Effect,caus.Predicted_Main_Effect),
            Additive_Sign_Accuracy=float(np.mean(np.sign(caus.True_Main_Effect)==np.sign(caus.Predicted_Main_Effect))),
            Additive_Effect_MAE=float(np.mean(np.abs(caus.Effect_Error))),
            Additive_Precision_AUPRC=float(average_precision_score(y,score)),
            Additive_Recovery_AUROC=float(roc_auc_score(y,score)))
    if pairs is not None:
        caus=pairs[pairs.Is_Known_Interaction==1];y=pairs.Is_Known_Interaction.to_numpy(int)
        score=pairs.Mean_Absolute_Interaction.to_numpy(float)
        res.update(Epistatic_N=len(caus),Epistatic_Effect_Corr=nan_r(caus.True_Interaction_Effect,caus.Predicted_Interaction_Effect),
            Epistatic_Sign_Accuracy=float(np.mean(np.sign(caus.True_Interaction_Effect)==np.sign(caus.Predicted_Interaction_Effect))),
            Epistatic_Effect_MAE=float(np.mean(np.abs(caus.Effect_Error))),
            Epistatic_Precision_AUPRC=float(average_precision_score(y,score)),
            Epistatic_Recovery_AUROC=float(roc_auc_score(y,score)))
    return res


def check_config(c):
    if c['latent_dim']!=128:raise ValueError('This is the 128-D original PerfVAE manuscript')
    if not c['scenarios_to_run'] or any(s not in c['scenarios'] for s in c['scenarios_to_run']):
        raise ValueError('Invalid selected scenarios')
    if c['folds']!=10 or not c['folds_to_run'] or max(c['folds_to_run'])>c['folds'] or min(c['folds_to_run'])<1:
        raise ValueError('Keep original 10-fold split; folds_to_run is subset of 1..10')
    if not c['controlled_marker_counts'] or len(set(c['controlled_marker_counts']))!=len(c['controlled_marker_counts']):
        raise ValueError('Need unique marker counts')
    if any(m<c['controlled_additive_snps']+2*c['controlled_epistatic_pairs'] for m in c['controlled_marker_counts']):
        raise ValueError('Panel too small for all causal SNPs')
    if c['max_test_backgrounds']<1 or c['min_pair_joint_count']<1:raise ValueError('Invalid support settings')


def summarize(root):
    raw=root/'fold_results.csv'
    if not raw.exists():return
    tab=pd.read_csv(raw)
    sums=[]
    for (panel,scenario),grp in tab.groupby(['Panel','Scenario'],sort=True):
        row=dict(Panel=panel,Scenario=scenario,Completed_Folds=len(grp),
                 Model='Original_Full_PerfVAE_128D')
        for k in ['Test_Pearson','Test_R2','Additive_Effect_Corr','Additive_Sign_Accuracy',
                  'Additive_Effect_MAE','Additive_Precision_AUPRC','Additive_Recovery_AUROC',
                  'Epistatic_Effect_Corr','Epistatic_Sign_Accuracy','Epistatic_Effect_MAE',
                  'Epistatic_Precision_AUPRC','Epistatic_Recovery_AUROC']:
            vals=pd.to_numeric(grp[k],errors='coerce') if k in grp else pd.Series(dtype=float)
            row[k+'_mean']=float(vals.mean()) if vals.notna().any() else np.nan
            row[k+'_std']=float(vals.std(ddof=1)) if vals.notna().sum()>1 else np.nan
        sums.append(row)
    atomic_csv(pd.DataFrame(sums),root/'all_tasks_summary.csv')



def make_effect_plots(root,panel,c):
    if not c.get('make_plots',False):return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not installed; skipping optional effect plots',flush=True)
        return
    sub=root/panel
    additive=[];interactions=[]
    for fold in c['folds_to_run']:
        for scenario in c['scenarios_to_run']:
            if scenario in ('additive','mixed'):
                path=sub/f'additive_effects_{scenario}_fold{fold}.csv'
                if path.exists():additive.append(pd.read_csv(path))
            if scenario in ('epistatic','mixed'):
                path=sub/f'interaction_effects_{scenario}_fold{fold}.csv'
                if path.exists():interactions.append(pd.read_csv(path))
    for sc in sorted(set(df.Scenario.iloc[0] for df in additive)):
        tab=pd.concat([df for df in additive if df.Scenario.iloc[0]==sc],ignore_index=True)
        caus=tab[tab.Is_Known_Additive==1]
        ave=caus.groupby('Original_SNP_Index_0Based',as_index=False)[['True_Main_Effect','Predicted_Main_Effect']].mean()
        fig,ax=plt.subplots(figsize=(5.5,5))
        ax.scatter(ave.True_Main_Effect,ave.Predicted_Main_Effect)
        lo=min(ave.True_Main_Effect.min(),ave.Predicted_Main_Effect.min())
        hi=max(ave.True_Main_Effect.max(),ave.Predicted_Main_Effect.max())
        ax.plot([lo,hi],[lo,hi],linestyle='--',linewidth=1)
        ax.set(xlabel='True additive SNP contrast',ylabel='PerfVAE estimated contrast',
               title=f'{panel}: {sc}, {len(ave)} causal SNPs')
        fig.tight_layout();fig.savefig(sub/f'Figure_Additive_Effect_Recovery_{sc}.png',dpi=180)
        plt.close(fig)
    for sc in sorted(set(df.Scenario.iloc[0] for df in interactions)):
        tab=pd.concat([df for df in interactions if df.Scenario.iloc[0]==sc],ignore_index=True)
        group=tab[tab.Is_Known_Interaction==1].groupby(['SNP1_Original_Index_0Based',
                  'SNP2_Original_Index_0Based'],as_index=False)[['True_Interaction_Effect',
                  'Predicted_Interaction_Effect']].mean()
        fig,ax=plt.subplots(figsize=(5.5,5))
        ax.scatter(group.True_Interaction_Effect,group.Predicted_Interaction_Effect)
        lo=min(group.True_Interaction_Effect.min(),group.Predicted_Interaction_Effect.min())
        hi=max(group.True_Interaction_Effect.max(),group.Predicted_Interaction_Effect.max())
        ax.plot([lo,hi],[lo,hi],linestyle='--',linewidth=1)
        ax.set(xlabel='True SNP-pair interaction contrast',ylabel='PerfVAE estimated contrast',
               title=f'{panel}: {sc}, {len(group)} causal SNP pairs')
        fig.tight_layout();fig.savefig(sub/f'Figure_Epistatic_Effect_Recovery_{sc}.png',dpi=180)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,4))
        neg=tab.loc[tab.Is_Known_Interaction==0,'Mean_Absolute_Interaction'].to_numpy()
        pos=tab.loc[tab.Is_Known_Interaction==1,'Mean_Absolute_Interaction'].to_numpy()
        try:
            ax.boxplot([neg,pos],tick_labels=['No simulated pair effect','Known interacting pair'])
        except TypeError:
            ax.boxplot([neg,pos],labels=['No simulated pair effect','Known interacting pair'])
        ax.set(ylabel='Mean absolute interaction contrast',
               title=f'{panel}: {sc} predicted interaction strengths')
        fig.tight_layout();fig.savefig(sub/f'Figure_Epistatic_Negative_Control_{sc}.png',dpi=180)
        plt.close(fig)

def main():
    c=dict(CONFIG)
    check_config(c)
    if Performer is None:raise ImportError('Install performer-pytorch: python -m pip install performer-pytorch')
    X,src=load_wheat2000(c)
    digest=sha256_file(src)
    if X.shape!=(2000,33709):
        print(f'WARNING: Wheat2000 X={X.shape}; expected (2000,33709)',flush=True)
    device=torch.device(c['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=bool(c['tf32']);torch.backends.cudnn.allow_tf32=bool(c['tf32'])
    seed_everything(c['seed'])
    Y,G,causal,report=controlled_simulation(X,c)
    truth=make_truth_spec(X,causal,c)
    for s in Y:
        actual=math.sqrt(c['heritability'])*(
            (math.sqrt(1-c['scenarios'][s])*truth['A_std']+math.sqrt(c['scenarios'][s])*truth['I_std'])/
            truth['weighted_sd'][s])
        if not np.allclose(actual,G[s],atol=1e-6):
            raise AssertionError(f'Analytical effect truth mismatch against simulator for {s}')
    panels=nested_marker_panels(X,c,causal)
    root=Path(c['out_dir']);root.mkdir(parents=True,exist_ok=True)
    atomic_csv(report,root/'simulation_variance_report.csv')
    for k,v in causal.items():atomic_csv(v,root/(k+'.csv'))
    atomic_csv(pd.DataFrame({'Sample_Row_Index_0Based':np.arange(len(X)),
             **{f'Simulated_{q}':vv for q,vv in Y.items()},
             **{f'True_Genetic_{q}':vv for q,vv in G.items()}}),
             root/'controlled_simulated_phenotypes.csv')
    print('='*80,flush=True)
    print('Wheat2000 | ORIGINAL 128D FULL PerfVAE | KNOWN additive+epistatic effect recovery',flush=True)
    print(f'Genotype={src} | X={X.shape} | 10-fold seed={c["seed"]}, run folds={c["folds_to_run"]}',flush=True)
    print(f'C={c["chunk_size"]}, Epochs={c["epochs"]}, scenarios={c["scenarios_to_run"]}, panels={c["controlled_marker_counts"]}',flush=True)
    print('='*80,flush=True)
    kf=list(KFold(n_splits=c['folds'],shuffle=True,random_state=c['seed']).split(X))
    assignment=np.zeros(len(X),dtype=int)
    for num,(_tr,te) in enumerate(kf,1):assignment[te]=num
    atomic_csv(pd.DataFrame({'Sample_Row_Index_0Based':np.arange(len(X)),
           'Heldout_Fold_1Based':assignment}),root/'cv_fold_assignments.csv')
    for n,ids in panels.items():
        panel=f'controlled_{n}'
        sig,metadata=make_signature(c,digest,ids)
        pr=root/panel;pr.mkdir(parents=True,exist_ok=True)
        settings=pr/'settings_and_provenance.json'
        if settings.is_file():
            old=json.loads(settings.read_text(encoding='utf8'))
            if old['Config_ID']!=sig:raise ValueError(f'Existing {panel} result configuration differs: use new out_dir')
        else:atomic_json({'Config_ID':sig,**metadata},settings)
        atomic_csv(pd.DataFrame({'Original_SNP_Index_0Based':ids,
                   'Panel_SNP_Index_0Based':np.arange(len(ids))}),pr/'selected_snp_manifest.csv')
        Xp=np.ascontiguousarray(X[:,ids])
        mapping={int(orig):k for k,orig in enumerate(ids)}
        done={(str(r.Panel),str(r.Scenario),int(r.Fold)) for r in pd.read_csv(root/'fold_results.csv').itertuples() if str(r.Config_ID)==sig} if (root/'fold_results.csv').exists() else set()
        if (root/'fold_results.csv').exists():
            allres=pd.read_csv(root/'fold_results.csv')
            bad=allres[(allres.Panel==panel)&(allres.Config_ID!=sig)]
            if len(bad):raise ValueError('Existing fold_results.csv signature mismatch')
        else:allres=pd.DataFrame()
        print(f'PANEL {panel} X={Xp.shape}',flush=True)
        for fold,(tr,te) in enumerate(kf,1):
            if fold not in c['folds_to_run']:continue
            for scenario in c['scenarios_to_run']:
                if (panel,scenario,fold) in done:
                    print(f'  {scenario} fold{fold} complete; skip',flush=True);continue
                t0=time.perf_counter()
                trainx=Xp[tr];testx=Xp[te]
                proc_tr,proc_te,transform=fitted_x_preprocess(trainx,testx)
                scaler=raw_y_scaler(Y[scenario][tr])
                trainy=np.asarray(scaler.transform(Y[scenario][tr,None]),dtype=np.float32)
                testy=np.asarray(scaler.transform(Y[scenario][te,None]),dtype=np.float32)
                gxtr=torch.from_numpy(proc_tr).to(device)
                gxte=torch.from_numpy(proc_te).to(device)
                gytr=torch.from_numpy(trainy).to(device)
                gyte=torch.from_numpy(testy).to(device)
                ckpt=pr/'checkpoints'/f'{scenario}_fold{fold}.pth'
                model=None
                if c['resume'] and ckpt.exists():
                    state=torch.load(ckpt,map_location='cpu',weights_only=False)
                    if state.get('Config_ID')!=sig:raise ValueError(f'Checkpoint mismatched: {ckpt}')
                    model=make_nn('full_perfvae',n,c).to(device)
                    model.load_state_dict(state['state_dict'])
                    elapsed=float(state.get('Training_Time_s',np.nan))
                    print(f'  [{scenario}] Fold {fold} checkpoint loaded',flush=True)
                else:
                    print(f'  [{scenario}] Fold {fold}: TRAIN original Full PerfVAE',flush=True)
                    output=run_nn('full_perfvae',gxtr,gytr,gxte,gyte,c,c['seed']+fold,device)
                    model=output['model'];elapsed=float(output['time_s'])
                    if c['save_checkpoints']:
                        ckpt.parent.mkdir(parents=True,exist_ok=True)
                        tmp=ckpt.with_suffix('.tmp')
                        torch.save({'Config_ID':sig,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                                    'Training_Time_s':elapsed},tmp)
                        os.replace(tmp,ckpt)
                    del output
                with torch.inference_mode():
                    pred_tr=predict_nn(model,gxtr,c['batch_size']).detach().cpu().numpy()
                    pred_te=predict_nn(model,gxte,c['batch_size']).detach().cpu().numpy()
                pear=nan_r(gyte.cpu().numpy(),pred_te)
                trainr=nan_r(gytr.cpu().numpy(),pred_tr)
                yr=gyte.cpu().numpy().reshape(-1); yp=pred_te.reshape(-1)
                r2=float(1-np.mean((yp-yr)**2)/max(float(np.mean((yr-yr.mean())**2)),1e-12))
                backgrounds=testx[:int(c['max_test_backgrounds'])]
                mains=None;pairs=None
                if scenario in ('additive','mixed'):
                    mains=assess_main_effects(model,trainx,backgrounds,ids,truth,c,scenario,
                        mapping,transform,scaler,device,fold,panel,c['seed']+fold)
                    mains.insert(0,'Config_ID',sig)
                    atomic_csv(mains,pr/f'additive_effects_{scenario}_fold{fold}.csv')
                if scenario in ('epistatic','mixed'):
                    pairs=assess_pair_effects(model,trainx,backgrounds,ids,truth,c,scenario,
                        mapping,transform,scaler,device,fold,panel,c['seed']+fold)
                    pairs.insert(0,'Config_ID',sig)
                    atomic_csv(pairs,pr/f'interaction_effects_{scenario}_fold{fold}.csv')
                summary=effect_summary(mains,pairs,c,fold,scenario,panel,pear,r2,trainr,elapsed,
                        sum(p.numel() for p in model.parameters()),sig)
                allres=pd.concat([allres,pd.DataFrame([summary])],ignore_index=True)
                atomic_csv(allres,root/'fold_results.csv');summarize(root)
                print(f'    Fold {fold} {scenario} | test r={pear:.4f} R2={r2:.4f} | '
                      f'Main corr={summary.get("Additive_Effect_Corr",float("nan")):.3f} | '
                      f'Epi AUPRC={summary.get("Epistatic_Precision_AUPRC",float("nan")):.3f} | '
                      f'fit={elapsed:.1f}s | total={time.perf_counter()-t0:.1f}s',flush=True)
                del model,gxtr,gxte,gytr,gyte
        make_effect_plots(root,panel,c)
        print('Panel complete/partial:',pr,flush=True)
    print('SUMMARY:',root/'all_tasks_summary.csv',flush=True)

if __name__=='__main__':main()
