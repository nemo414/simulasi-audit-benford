#!/usr/bin/env python3
import json
import math
import os
import time

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_final")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# PARAMETER 
# ----------------------------------------------------------------------------
P = dict(
    N=5000,            # jumlah wajib pajak sintetis per periode
    REPS=300,          # ulangan Monte Carlo
    TPER=8,            # jumlah periode pajak per ulangan
    PI=0.15,           # proporsi wajib pajak yang menghindar pajak
    SHARE_DET=0.6,     # porsi pelanggar yang memalsukan angka (meninggalkan jejak digit)
    M_LO=0.2, M_HI=0.6,  # porsi angka yang dipalsukan ~ U(M_LO, M_HI)
    N_MED=400, N_SIG=0.6, N_MIN=150, N_MAX=2000,   # jumlah faktur per wajib pajak
    T_MED=100.0, T_SIG=1.0,                        # pajak terutang (juta Rp) ~ lognormal
    DEL_LO=0.10, DEL_HI=0.40,                      # porsi pajak yang dihindari ~ U
    F=1.0,             # tarif denda (kelipatan pajak yang dihindari)
    C=5.0,             # biaya satu pemeriksaan (juta Rp)
    S=0.8,             # peluang audit menemukan pelanggaran
    EPS=0.10,          # porsi anggaran audit yang diacak pada strategi terintegrasi
    PRIOR_A=1.0, PRIOR_B=9.0,   # prior Beta untuk peluang temuan per kelompok
    BUDGETS=[0.005, 0.01, 0.02, 0.05, 0.10],      # anggaran audit (porsi dari N)
    MAIN_BUDGET=0.01,
    PHI_STEP=0.10, PHI_MAX=0.70,   # laju pelanggar "sadar Benford" per periode
    SEED=20261026,
    NB_SHARE=0.0,      # porsi wajib pajak PATUH dengan digit non-Benford alami (uji tekanan)
    FAB="seragam",     # pola pemalsuan: "seragam" atau "tengah" (angka menengah dilebih-lebihkan)
)
DELTA_BAR = (P["DEL_LO"] + P["DEL_HI"]) / 2.0
BENFORD = np.log10(1 + 1 / np.arange(1, 10))
UNIF = np.full(9, 1 / 9)
CHI_5 = float(stats.chi2.ppf(0.95, 8))
CHI_01 = float(stats.chi2.ppf(0.999, 8))
STRATS = ["acak", "skor", "ukuran", "dua_tahap", "int_eg", "int_ts"]
STRATS_MAIN = ["acak", "skor", "ukuran", "dua_tahap", "int_eg", "int_ts"]   # digambar di grafik
STRAT_LABEL = {"acak": "Audit acak", "skor": "Skor Benford saja",
               "ukuran": "Ukuran pajak saja", "dua_tahap": "Dua tahap (χ² lalu ukuran)",
               "int_eg": "Terintegrasi (ε-greedy)", "int_ts": "Terintegrasi (Thompson)"}
NB_PROBS = np.array([0.10, 0.08, 0.08, 0.08, 0.10, 0.10, 0.12, 0.14, 0.20])  # harga psikologis/pembulatan
FAB_PROBS = {"seragam": UNIF,
             "tengah": np.array([0.06, 0.09, 0.12, 0.14, 0.16, 0.15, 0.12, 0.09, 0.07])}


# ----------------------------------------------------------------------------
# Populasi sintetis
# ----------------------------------------------------------------------------
def draw_base(rng):
    """Bilangan acak dasar satu periode; dipakai bersama (common random numbers) oleh semua strategi."""
    N = P["N"]
    return dict(
        n=np.clip(np.round(rng.lognormal(np.log(P["N_MED"]), P["N_SIG"], N)),
                  P["N_MIN"], P["N_MAX"]).astype(np.int64),
        T=rng.lognormal(np.log(P["T_MED"]), P["T_SIG"], N),
        u_ev=rng.random(N), u_det=rng.random(N), u_aw=rng.random(N), m=rng.uniform(P["M_LO"], P["M_HI"], N),
        delta=rng.uniform(P["DEL_LO"], P["DEL_HI"], N), u_caught=rng.random(N), u_nb=rng.random(N))


def multinom_rows(rng, cnt, probs, mask):
    out = np.zeros((len(cnt), 9), dtype=np.int64)
    if mask.any():
        out[mask] = rng.multinomial(cnt[mask], probs)
    return out


def make_population(rng, phi, base=None, sc_det=1.0, sc_oth=1.0, rng_dig=None):
    """sc_det / sc_oth: pengali peluang menghindar (respons jera) untuk tipe pemalsu / bukan pemalsu."""
    if base is None:
        base = draw_base(rng)
    if rng_dig is None:
        rng_dig = rng
    n, T = base["n"], base["T"]
    det_type = base["u_det"] < P["SHARE_DET"]
    p_ev = P["PI"] * np.where(det_type, sc_det, sc_oth)
    evader = base["u_ev"] < p_ev
    detectable = evader & det_type
    aware = detectable & (base["u_aw"] < phi)         # sadar Benford: angka palsu ikut pola Benford
    trace = detectable & ~aware                       # meninggalkan jejak digit
    k = np.where(trace, rng_dig.binomial(n, base["m"]), 0)
    nbm = (~evader) & (base["u_nb"] < P["NB_SHARE"])  # patuh tetapi non-Benford alami
    cnt = (multinom_rows(rng_dig, n - k, BENFORD, ~nbm) + multinom_rows(rng_dig, n - k, NB_PROBS, nbm)
           + multinom_rows(rng_dig, k, FAB_PROBS[P["FAB"]], k > 0))
    E = n[:, None] * BENFORD[None, :]
    chi2 = ((cnt - E) ** 2 / E).sum(1)
    mad = np.abs(cnt / n[:, None] - BENFORD[None, :]).mean(1)
    d = np.where(evader, base["delta"] * T, 0.0)
    caught = evader & (base["u_caught"] < P["S"])
    return dict(n=n, T=T, evader=evader, detectable=detectable, trace=trace, aware=aware,
                chi2=chi2, mad=mad, d=d, caught=caught, cnt=cnt, nonben=nbm)


def tiers_from_chi2(chi2):
    return (chi2 >= CHI_5).astype(int) + (chi2 >= CHI_01).astype(int)   # 0 rendah, 1 sedang, 2 tinggi


# ----------------------------------------------------------------------------
# Strategi seleksi audit
# ----------------------------------------------------------------------------
def pick_random(rng, N, B):
    mask = np.zeros(N, bool)
    mask[rng.choice(N, B, replace=False)] = True
    return mask


def pick_score(chi2, B):
    mask = np.zeros(len(chi2), bool)
    mask[np.argpartition(-chi2, B - 1)[:B]] = True
    return mask












def new_state():
    return dict(a=np.full(3, P["PRIOR_A"]), b=np.full(3, P["PRIOR_B"]))


def update_state(st, mask, pop, tier):
    for k in range(3):
        mk = mask & (tier == k)
        hits = int((mk & pop["caught"]).sum())
        st["a"][k] += hits
        st["b"][k] += int(mk.sum()) - hits


def evaluate(mask, pop):
    B = int(mask.sum())
    caught = mask & pop["caught"]
    det = int(caught.sum())
    rec = float((pop["d"][caught] * (1 + P["F"])).sum())
    return det, rec, rec - P["C"] * B



# Koreksi final: posterior terpotong, RNG seleksi per strategi, dan holdout seed.
from scipy.special import betainc, betaincinv
import hashlib, platform, sys

def bounded_mean(a,b):
    z=betainc(a,b,P['S'])
    return a/(a+b)*betainc(a+1,b,P['S'])/z

def bounded_draw(rng,a,b):
    z=betainc(a,b,P['S'])
    r=betaincinv(a,b,rng.random(3)*z)
    assert np.all((r>=0)&(r<=P['S']+1e-12))
    return r

def ci(v):
    v=np.asarray(v,float)
    return [float(v.mean()),float(1.96*v.std(ddof=1)/np.sqrt(len(v)))]

def rng_for(seed,rep,key):
    keynum=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
    return np.random.default_rng([seed,rep,keynum])

def policy(key,rng,pop,tier,B,st,chosen):
    T=pop.get('proxy',pop['T'])
    if key=='acak':return pick_random(rng,len(T),B)
    if key=='benford':return pick_score(pop['chi2'],B)
    if key=='ukuran':
        m=np.zeros(len(T),bool);m[np.argsort(-T)[:B]]=True;return m
    if key.startswith('two_') or key=='two_best':
        K,eps=chosen if key=='two_best' else (int(key.split('_')[1]),float(key.split('_')[2]))
        nc=B-int(round(eps*B));cand=np.argsort(-pop['chi2'])[:min(len(T),K*B)]
        m=np.zeros(len(T),bool);m[cand[np.argsort(-T[cand])[:nc]]]=True
        need=B-int(m.sum())
        if need:m[rng.choice(np.flatnonzero(~m),need,replace=False)]=True
        return m
    if key=='ts_raw':r=rng.beta(st['a'],st['b']);eps=0
    elif key in ['ts','ts_disc','ts10']:r=bounded_draw(rng,st['a'],st['b']);eps=.1 if key=='ts10' else 0
    else:r=bounded_mean(st['a'],st['b']);eps=.1
    values=r[tier]*DELTA_BAR*T*(1+P['F'])-P['C']
    nmax=B-int(round(eps*B));idx=np.argsort(-values)[:nmax];idx=idx[values[idx]>0]
    m=np.zeros(len(T),bool);m[idx]=True
    need=B-int(m.sum())
    if need:m[rng.choice(np.flatnonzero(~m),need,replace=False)]=True
    return m

def experiment(keys,reps,seed,share=.01,chosen=(4,0),phi=False,kappa=0,nb=0,trace=.6,noise=0,rho=0,ablate=False):
    """Populasi dan hasil deteksi potensial berpasangan. Parameter tidak disetel memakai holdout."""
    oldnb,oldtrace=P['NB_SHARE'],P['SHARE_DET'];P['NB_SHARE']=nb;P['SHARE_DET']=trace
    R,TP=reps,P['TPER'];B=round(P['N']*share)
    raw={k:{x:np.zeros((R,TP)) for x in ['net','det','viol','gap','cover_low','fiscal_margin','audit_compliant']} for k in keys}
    auc=[];fpr=[]
    for rep in range(R):
        states={k:new_state() for k in keys};prev={k:(0.,0.) for k in keys}
        rsel={k:rng_for(seed+10,rep,k) for k in keys}
        for t in range(TP):
            base=draw_base(np.random.default_rng([seed,rep,t]))
            if rho:
                z=np.log(base['T']/P['T_MED'])/P['T_SIG']
                # Probit: marginal PI tetap bila z normal; korelasi laten rho.
                prob=stats.norm.cdf((stats.norm.ppf(P['PI'])+rho*z)/np.sqrt(1-rho*rho))
                base['u_ev']=base['u_ev']*P['PI']/prob
            common_pop=None
            for key in keys:
                low,high=prev[key]
                if not kappa and common_pop is not None:pop=common_pop
                else:
                    pop=make_population(None,min(.7,.1*t) if phi else 0,base=base,
                        sc_det=np.exp(-kappa*high),sc_oth=np.exp(-kappa*low),
                        rng_dig=np.random.default_rng([seed+1,rep,t]))
                    if noise:
                        znoise=np.random.default_rng([seed+2,rep,t]).normal(0,noise,P['N'])
                        pop['proxy']=pop['T']*np.exp(znoise-noise*noise/2)
                    if not kappa:common_pop=pop
                tier=tiers_from_chi2(pop['chi2']);st=states[key]
                if t==0 and key==keys[0]:
                    rk=stats.rankdata(pop['chi2']);ev=pop['evader'];n1=int(ev.sum());n0=len(ev)-n1
                    auc.append(float((rk[ev].sum()-n1*(n1+1)/2)/(n1*n0)))
                    fpr.append(float((pop['chi2'][~ev]>=CHI_5).mean()))
                if ablate and t==0:
                    m=pick_random(rng_for(seed+30,rep,'warmup'),P['N'],B)
                else:m=policy(key,rsel[key],pop,tier,B,st,chosen)
                dd,rc,nt=evaluate(m,pop)
                raw[key]['net'][rep,t]=nt;raw[key]['det'][rep,t]=dd
                raw[key]['viol'][rep,t]=pop['evader'].mean()
                raw[key]['gap'][rep,t]=pop['d'].sum()
                # Total pajak bersih = sum(T) - gap + net. sum(T) sama antarstrategi,
                # sehingga selisih fiskal berpasangan cukup memakai net - gap.
                raw[key]['fiscal_margin'][rep,t]=nt-pop['d'].sum()
                raw[key]['audit_compliant'][rep,t]=(m&~pop['evader']).sum()/B
                lo,hi=tier==0,tier==2
                prev[key]=((m&lo).sum()/max(1,lo.sum()),(m&hi).sum()/max(1,hi.sum()))
                raw[key]['cover_low'][rep,t]=prev[key][0]
                learn=key in ['ts','ts10','ts_disc','eg','ts_raw'] or (key=='fixed' and t==0)
                if learn:
                    if key=='ts_disc':
                        st['a']=P['PRIOR_A']+.9*(st['a']-P['PRIOR_A'])
                        st['b']=P['PRIOR_B']+.9*(st['b']-P['PRIOR_B'])
                    update_state(st,m,pop,tier)
    P['NB_SHARE'],P['SHARE_DET']=oldnb,oldtrace
    out={'reps':reps,'seed':seed,'B':B,'auc':ci(auc),'fpr':ci(fpr),'strategies':{},'differences':{},'fiscal_differences':{}}
    for key in keys:
        out['strategies'][key]={m:{'mean':ci(v.mean(axis=1)),'last':ci(v[:,-1]),'period':v.mean(axis=0).tolist()} for m,v in raw[key].items()}
        out['strategies'][key]['precision']=float(raw[key]['det'].mean()/B)
    for a in keys:
        for b in keys:
            if a!=b:
                out['differences'][a+'__'+b]=ci(raw[a]['net'].mean(1)-raw[b]['net'].mean(1))
                out['fiscal_differences'][a+'__'+b]=ci(raw[a]['fiscal_margin'].mean(1)-raw[b]['fiscal_margin'].mean(1))
    return out

def main_final():
    start=time.time()
    print('Seleksi pembanding pada seed pengembangan...',flush=True)
    candidates=[f'two_{k}_{e:g}' for k in [2,4,8] for e in [0,.1]]
    dev=experiment(candidates,100,20261030)
    best=max(candidates,key=lambda k:dev['strategies'][k]['net']['mean'][0])
    chosen=(int(best.split('_')[1]),float(best.split('_')[2]))
    print('Terpilih',chosen,flush=True)
    keys=['acak','benford','ukuran','two_best','eg','ts']
    results={'parameters':P.copy(),'protocol':{'development_seed':20261030,'validation_seed':20261031,'chosen_two_stage':chosen,'candidates':candidates,'units':'juta rupiah per periode','CI':'mean +/- 1.96 SE antarulangan; selisih berpasangan; eksploratif tanpa koreksi multipel'},'development':dev}
    results['main']=experiment(keys,300,20261031,chosen=chosen)
    print('Utama selesai',round(time.time()-start),flush=True)
    results['budget']={}
    for sh in [.005,.02,.05,.1]:
        results['budget'][str(sh)]=experiment(['two_best','ts'],100,20261031,share=sh,chosen=chosen)
    results['ablation']=experiment(['fixed','eg','ts10','ts'],300,20261101,chosen=chosen,ablate=True)
    results['bounded_check']=experiment(['ts_raw','ts'],300,20261102,chosen=chosen)
    print('Ablasi dan koreksi posterior selesai',round(time.time()-start),flush=True)
    scenarios={'dasar':{},'proksi_bising':{'noise':.5},'risiko_terkait_ukuran':{'rho':.5},
        'gabungan':{'noise':.5,'nb':.2,'trace':.4,'rho':.5},'nonbenford20':{'nb':.2},'nonbenford50':{'nb':.5}}
    results['stress']={}
    for name,kw in scenarios.items():
        results['stress'][name]=experiment(['ukuran','two_best','ts'],100,20261103,chosen=chosen,**kw)
        print('Tekanan',name,flush=True)
    results['adaptation']=experiment(['two_best','ts','ts_disc'],150,20261104,chosen=chosen,phi=True)
    results['deterrence']={str(k):experiment(['acak','benford','ukuran','two_best','ts'],100,20261105,chosen=chosen,kappa=k) for k in [0,5,15]}
    results['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'matplotlib':matplotlib.__version__}
    Path=__import__('pathlib').Path
    (Path(OUT)/'hasil_final.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    from ekspor_hasil import export_results
    export_results(results, OUT)
    print('SELESAI',round(time.time()-start),'detik',flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hanya-output',action='store_true',
                        help='Buat tabel CSV dan grafik PNG dari JSON yang sudah tersedia; tanpa simulasi ulang.')
    args=parser.parse_args()
    if args.hanya_output:
        from pathlib import Path
        from ekspor_hasil import export_results
        export_results(json.loads((Path(OUT)/'hasil_final.json').read_text(encoding='utf-8')),OUT)
    else:
        main_final()
