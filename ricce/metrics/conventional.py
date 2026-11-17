from collections import Counter
import numpy as np

def _tokens(s): return s.lower().split()

def sentence_bleu_scores(pred,refs):
    try:
        from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
        rt=[_tokens(r) for r in refs]; pt=_tokens(pred); sm=SmoothingFunction().method1; out={}
        for n in range(1,5):
            w=tuple([1/n]*n+[0]*(4-n)); out[f'bleu{n}']=float(sentence_bleu(rt,pt,weights=w,smoothing_function=sm))
        return out
    except Exception:return {f'bleu{i}':float('nan') for i in range(1,5)}

def rouge_l(pred,refs):
    try:
        from rouge_score import rouge_scorer
        sc=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True); vals=[sc.score(r,pred)['rougeL'].fmeasure for r in refs]
        return max(vals) if vals else float('nan')
    except Exception:return float('nan')

def meteor(pred,refs):
    try:
        from nltk.translate.meteor_score import meteor_score
        vals=[meteor_score([_tokens(r)],_tokens(pred)) for r in refs]; return max(vals) if vals else float('nan')
    except Exception:return float('nan')

def cider_diagnostic(pred,refs,n=4):
    """NOT official CIDEr-D -- a single-pair n-gram cosine-similarity proxy
    with no corpus-level IDF weighting. Kept only as an internal debugging
    signal; use cider_d_corpus for anything reported as CIDEr in a table
    (internal design review Sec. 6.1/7.7)."""
    if not refs:return float('nan')
    def ng(tok,k):return Counter(tuple(tok[i:i+k]) for i in range(len(tok)-k+1))
    scores=[]; pt=_tokens(pred)
    for r in refs:
        rt=_tokens(r); sims=[]
        for k in range(1,n+1):
            a,b=ng(pt,k),ng(rt,k); keys=set(a)|set(b); va=np.array([a[x] for x in keys],float); vb=np.array([b[x] for x in keys],float); den=np.linalg.norm(va)*np.linalg.norm(vb); sims.append(float(va.dot(vb)/den) if den else 0.0)
        scores.append(np.mean(sims))
    return float(np.mean(scores))*10.0

def cider_d_corpus(image_ids,predictions,references_by_id):
    """Official CIDEr-D (Vedantam et al. 2015), computed over the whole
    corpus at once as CIDEr requires (per-n-gram document frequency is a
    corpus-level statistic, unlike BLEU/METEOR/ROUGE-L). Returns
    (corpus_mean, {image_id: per_image_score})."""
    if not image_ids:return float('nan'),{}
    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception:
        return float('nan'),{i:float('nan') for i in image_ids}
    gts={i:references_by_id[i] for i in image_ids}
    res={i:[predictions[j]] for j,i in enumerate(image_ids)}
    overall,per_item=Cider().compute_score(gts,res)
    return float(overall),{i:float(s) for i,s in zip(image_ids,per_item)}

_SBERT_CACHE={}
def sentence_bert(pred,refs,model_name='sentence-transformers/all-MiniLM-L6-v2'):
    try:
        if not refs:return float('nan')
        if model_name not in _SBERT_CACHE:
            from sentence_transformers import SentenceTransformer
            _SBERT_CACHE[model_name]=SentenceTransformer(model_name,device='cpu')
        m=_SBERT_CACHE[model_name]
        e=m.encode([pred]+refs,normalize_embeddings=True); vals=e[1:]@e[0]; return float(np.max(vals))
    except Exception:return float('nan')

_BERTSCORER_CACHE={}
def bertscore(pred,refs,lang='en'):
    try:
        if not refs:return float('nan')
        if lang not in _BERTSCORER_CACHE:
            from bert_score import BERTScorer
            _BERTSCORER_CACHE[lang]=BERTScorer(lang=lang,rescale_with_baseline=False,device='cpu')
        scorer=_BERTSCORER_CACHE[lang]
        _,_,f=scorer.score([pred]*len(refs),refs)
        return float(f.max())
    except Exception:return float('nan')

def evaluate_caption(pred,refs,expensive=False):
    d=sentence_bleu_scores(pred,refs); d['meteor']=meteor(pred,refs); d['rougeL']=rouge_l(pred,refs); d['cider_diagnostic']=cider_diagnostic(pred,refs)
    if expensive:d['sentence_bert']=sentence_bert(pred,refs); d['bertscore']=bertscore(pred,refs)
    return d
