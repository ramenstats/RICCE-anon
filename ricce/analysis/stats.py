import numpy as np
from scipy import stats

def bootstrap_ci(values,func=np.mean,n_boot=10000,seed=42,alpha=0.05):
    x=np.asarray(values,float); x=x[np.isfinite(x)]
    if len(x)==0:return (float('nan'),float('nan'))
    rng=np.random.default_rng(seed); vals=[func(rng.choice(x,size=len(x),replace=True)) for _ in range(n_boot)]
    return tuple(np.quantile(vals,[alpha/2,1-alpha/2]).tolist())

def correlations(metric_values,human_values):
    x=np.asarray(metric_values,float); y=np.asarray(human_values,float); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<3:return {}
    return {'pearson_r':float(stats.pearsonr(x,y).statistic),'spearman_rho':float(stats.spearmanr(x,y).statistic),'kendall_tau':float(stats.kendalltau(x,y).statistic),'n':int(len(x))}

def paired_bootstrap_difference(a,b,n_boot=10000,seed=42):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]; rng=np.random.default_rng(seed); diffs=[]
    for _ in range(n_boot):
        idx=rng.integers(0,len(a),len(a)); diffs.append(float(np.mean(a[idx]-b[idx])))
    diffs=np.asarray(diffs); lo,hi=np.quantile(diffs,[0.025,0.975])
    p=float(min(1.0,2*min((diffs<=0).mean(),(diffs>=0).mean())))
    return {'mean_diff':float(np.mean(a-b)),'ci95':[float(lo),float(hi)],'p_value':p,'n':int(len(a))}

def wilcoxon_signed_rank(a,b):
    """Paired non-parametric significance test, complementing the bootstrap
    p-value above (internal design review, Phase I2)."""
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<2 or np.allclose(a,b):return {'statistic':float('nan'),'p_value':float('nan'),'n':int(len(a))}
    try:
        st=stats.wilcoxon(a,b); return {'statistic':float(st.statistic),'p_value':float(st.pvalue),'n':int(len(a))}
    except Exception:
        return {'statistic':float('nan'),'p_value':float('nan'),'n':int(len(a))}

def holm_correction(p_values):
    """Holm-Bonferroni step-down correction for a pre-declared family of
    planned comparisons (internal design review, Phase I3). Returns adjusted
    p-values in the ORIGINAL input order plus a reject-at-0.05 mask."""
    p=np.asarray(p_values,float); n=len(p); order=np.argsort(p)
    adj_sorted=np.empty(n,float); running_max=0.0
    for rank,idx in enumerate(order):
        val=(n-rank)*p[idx]; running_max=max(running_max,val); adj_sorted[rank]=min(1.0,running_max)
    adjusted=np.empty(n,float); adjusted[order]=adj_sorted
    return {'adjusted_p':adjusted.tolist(),'reject_at_0.05':(adjusted<0.05).tolist()}
