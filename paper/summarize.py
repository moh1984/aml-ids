"""Recomputes every number reported in the paper from the JSON files in results/.
Run from the repository root:  python paper/summarize.py
Outputs paper/numbers_static.json (Tables 5, 6, 11) and paper/numbers_adaptation.json (Tables 7-10).
Macro metrics are recomputed from the saved confusion matrices over the classes present in each test fold."""
# ---- static benchmarks, ablation, latency ----
import os,json,glob,numpy as np
from scipy.stats import ttest_rel
def cm_stats(cm):
    cm=np.array(cm,float); pres=cm.sum(1)>0
    tp=np.diag(cm); p=tp/np.maximum(cm.sum(0),1); rc=tp/np.maximum(cm.sum(1),1); f=2*p*rc/np.maximum(p+rc,1e-9)
    return dict(P=p[pres].mean()*100,R=rc[pres].mean()*100,F1=f[pres].mean()*100,f_cls=f*100,r_cls=rc*100,support=cm.sum(1))
S1='results/indomain'; S1b='results/indomain_rerun'; S2='results/indomain'; S4='results'
models=['svm','rf','cnn','lstm','cnn_lstm_attn','transformer','mamba','aml_ids']
out={'indomain':{}, 'stats':{}}
def load(d,m): return [json.load(open(f)) for f in sorted(glob.glob(f'{d}/{m}/fold*.json'))]
# UNSW-NB15 is evaluated on a randomly shuffled copy (random_state=0) because its files cluster records by class
FOLDER={'unsw_nb15':'unsw_shuffled'}
for ds,dirs in (('cicids2017',[S1,S1b]),('unsw_nb15',[S2]),('nslkdd',[S2])):
    out['indomain'][ds]={}; per={}
    for m in models:
        runs=[x for x in (load(d+'/'+FOLDER.get(ds,ds),m) for d in dirs) if x]   # the second CICIDS2017 run is optional
        # per-fold average over runs
        vals={k:[] for k in ['acc','P','R','F1','FPR','AUC','train']}
        for k in range(5):
            rs=[run[k] for run in runs]; c=[cm_stats(r['confusion']) for r in rs]
            vals['acc'].append(np.mean([r['accuracy'] for r in rs])); vals['FPR'].append(np.mean([r.get('fpr',np.nan) for r in rs]))
            vals['AUC'].append(np.mean([r.get('auc_roc',np.nan) for r in rs])); vals['train'].append(np.mean([r.get('train_min',np.nan) for r in rs]))
            for q in ['P','R','F1']: vals[q].append(np.mean([x[q] for x in c]))
        per[m]=np.array(vals['F1'])
        out['indomain'][ds][m]={k:(float(np.nanmean(v)),float(np.nanstd(v,ddof=1))) for k,v in vals.items()}
        cls=runs[0][0]['classes']
        fc=np.mean([cm_stats(r['confusion'])['f_cls'] for run in runs for r in run],0)
        out['indomain'][ds][m]['per_class_f1']=dict(zip(cls,[float(x) for x in fc]))
    out['stats'][ds]={}
    for m in models:
        if m=='aml_ids': continue
        t,p=ttest_rel(per['aml_ids'],per[m]); d=(per['aml_ids']-per[m]); 
        out['stats'][ds][m]=dict(diff=float(d.mean()),p=float(p),p_bonf=float(min(1,p*7)),d=float(d.mean()/d.std(ddof=1)))
# run-to-run
RUN_DIRS=[d for d in (S1,S1b) if glob.glob(f'{d}/cicids2017/aml_ids/fold*.json')]   # second run optional
out['rerun']={m:tuple(float(np.mean([cm_stats(r['confusion'])['F1'] for r in load(d+'/cicids2017',m)])) for d in RUN_DIRS) for m in models}
# supports
out['rows']={ds:int(sum(np.array(r['confusion']).sum() for r in load(d+'/'+ds,'rf'))) for ds,d in (('cicids2017',S1),('unsw_nb15',S2))}
nsl=load(S2+'/nslkdd','rf')[0]; out['rows']['nslkdd_test']=int(np.array(nsl['confusion']).sum())
out['classes']={ds:load(d+'/'+ds,'rf')[0]['classes'] for ds,d in (('cicids2017',S1),('unsw_nb15',S2),('nslkdd',S2))}
out['class_support_cic']=dict(zip(out['classes']['cicids2017'],[int(x) for x in sum(np.array(r['confusion']).sum(1) for r in load(S1+'/cicids2017','rf'))]))
# ablation
abl={}
Fa={}
for m in ['abl_base','abl_multiscale','abl_bilstm','abl_attention']:
    R=load(S4+'/indomain/cicids2017',m); v=np.array([cm_stats(r['confusion'])['F1'] for r in R]); Fa[m]=v
    abl[m]=dict(F1=(float(v.mean()),float(v.std(ddof=1))),acc=(float(np.mean([r['accuracy'] for r in R])),float(np.std([r['accuracy'] for r in R],ddof=1))),FPR=float(np.mean([r['fpr'] for r in R])),train=float(np.mean([r['train_min'] for r in R])))
Fa['aml_ids']=np.array([np.mean([cm_stats(load(d+'/cicids2017','aml_ids')[k]['confusion'])['F1'] for d in RUN_DIRS]) for k in range(5)])
seq=['abl_base','abl_multiscale','abl_bilstm','abl_attention','aml_ids']
abl['steps']=[dict(frm=a,to=b,diff=float((Fa[b]-Fa[a]).mean()),p=float(ttest_rel(Fa[b],Fa[a]).pvalue)) for a,b in zip(seq[:-1],seq[1:])]
out['ablation']=abl
# latency: average of the two runs
L1=json.load(open('results/latency/cicids2017.json'))
L2=json.load(open('results/latency_rerun/cicids2017.json')) if os.path.exists('results/latency_rerun/cicids2017.json') else L1
out['latency']={m:{k:(L1[m][k]+L2[m][k])/2 for k in L1[m]} for m in ['aml_ids','cnn_lstm_attn','transformer','mamba']}; out['latency']['ppf']=L1['packets_per_flow']
json.dump(out,open('paper/numbers_static.json','w'),indent=1,default=float)
for ds in out['indomain']:
    print(ds, {m:round(out['indomain'][ds][m]['F1'][0],1) for m in models})
print(out['rows'], out['class_support_cic'])
print(out['ablation']['steps'])

# ---- adaptation experiments ----
ms=lambda v:(float(np.mean(v)),float(np.std(v,ddof=1)))
out={}
L=[json.load(open(f)) for f in glob.glob('results/phases/cicids2017/lambda/*.json')+glob.glob('results/phases/cicids2017/lambda_large/*.json')]

# The configuration lam=0 without buffer exists in both lambda/ and lambda_large/ (identical deterministic runs of the
# same seeds). Keep one copy per (lambda, buffer, balanced, seed) so that each seed is counted once.
_seen = set(); _L = []
for r in L:
    k = (r['lam'], r['buffer'], r['balanced'], r['seed'])
    if k not in _seen:
        _seen.add(k); _L.append(r)
L = _L
lam={}
for r in L:
    lam.setdefault((r['lam'],r['buffer']),[]).append(r)
out['lambda']=[dict(lam=k[0],buffer=k[1],n=len(v),forg_f1=ms([r['forgetting_f1'] for r in v]),old_f1=ms([r['old_f1'] for r in v]),new_f1=ms([r['new_f1'] for r in v]),
   old_acc=ms([r['old_acc'] for r in v]),new_acc=ms([r['new_acc'] for r in v]),forg_acc=ms([r['forgetting'] for r in v]),new_before=ms([r['acc_before'][1] for r in v])) for k,v in sorted(lam.items())]
B=[json.load(open(f)) for f in glob.glob('results/phases/cicids2017/buffer/*.json')]
bb={}
for r in B: bb.setdefault((r['buffer'],r['balanced']),[]).append(r)
out['buffer']=[dict(buffer=k[0],balanced=k[1],forg_f1=ms([r['forgetting_f1'] for r in v]),old_f1=ms([r['old_f1'] for r in v]),new_f1=ms([r['new_f1'] for r in v]),new_acc=ms([r['new_acc'] for r in v]),old_acc=ms([r['old_acc'] for r in v]),adapt=ms([r['adapt_min'] for r in v])) for k,v in sorted(bb.items())]
un=[r['forgetting_f1'] for r in B if not r['balanced']]; ba=[r['forgetting_f1'] for r in B if r['balanced']]
# paired by (seed,buffer)
key=lambda r:(r['seed'],r['buffer'])
U={key(r):r['forgetting_f1'] for r in B if not r['balanced']}; Bl={key(r):r['forgetting_f1'] for r in B if r['balanced']}
ks=sorted(U); out['buffer_test']=dict(unbal=ms([U[k] for k in ks]),bal=ms([Bl[k] for k in ks]),p=float(ttest_rel([U[k] for k in ks],[Bl[k] for k in ks]).pvalue))
S=[json.load(open(f)) for f in glob.glob('results/stream/cicids2017/run/*.json')]
st={}
for r in S: st.setdefault(r['config'],[]).append(r)
def cmf1(cm):
    cm=np.array(cm,float); pres=cm.sum(1)>0; tp=np.diag(cm); p=tp/np.maximum(cm.sum(0),1); rc=tp/np.maximum(cm.sum(1),1); f=2*p*rc/np.maximum(p+rc,1e-9); return f[pres].mean()*100
out['stream']={c:dict(acc=ms([r['stream']['accuracy'] for r in v]),f1=ms([cmf1(r['stream']['confusion']) for r in v]),fpr=ms([r['stream']['fpr'] for r in v]),
   ret_acc=ms([r['retention']['accuracy'] for r in v]),ret_f1=ms([cmf1(r['retention']['confusion']) for r in v]),events=ms([r['drift_events'] for r in v]),adapt_min=ms([r['adapt_min'] for r in v]),
   stream_recall={k:float(np.mean([r['stream']['per_class_recall'].get(k,0) for r in v])) for k in v[0]['stream']['per_class_recall']},
   ret_recall={k:float(np.mean([r['retention']['per_class_recall'].get(k,0) for r in v])) for k in v[0]['retention']['per_class_recall']}) for c,v in st.items()}
st_support=np.array(S[0]['stream']['confusion']).sum(1); out['stream_support']=[int(x) for x in st_support]; out['stream_n']=int(st_support.sum())
# stream stats: replay vs naive retention DoS
def dos(c): return [r['retention']['per_class_recall']['DoS'] for r in sorted(st[c],key=lambda r:r['seed'])]
out['stream_p_replay_vs_naive_retF1']=float(ttest_rel([cmf1(r['retention']['confusion']) for r in sorted(st['ph:replay'],key=lambda r:r['seed'])],[cmf1(r['retention']['confusion']) for r in sorted(st['ph:naive'],key=lambda r:r['seed'])]).pvalue)
C=[json.load(open(f)) for f in sorted(glob.glob('results/cross/*/seed*.json'))]
out['cross']={'zero':dict(acc=ms([r['zero_shot']['accuracy'] for r in C]),f1=ms([r['zero_shot']['f1'] for r in C]),src=ms([r['source_before']['f1'] for r in C]))}
for v in ['replay','ewc','ewconly','naive']:
    out['cross'][v]=dict(acc=ms([r[f'adapted_{v}']['accuracy'] for r in C]),f1=ms([r[f'adapted_{v}']['f1'] for r in C]),src=ms([r[f'source_after_{v}']['f1'] for r in C]),sec=ms([r[f'adapt_sec_{v}'] for r in C]),fpr=ms([r[f'adapted_{v}']['fpr'] for r in C]))
out['cross']['p_gain']=float(ttest_rel([r['adapted_replay']['f1'] for r in C],[r['zero_shot']['f1'] for r in C]).pvalue)
out['cross']['p_ret']=float(ttest_rel([r['source_after_replay']['f1'] for r in C],[r['source_after_naive']['f1'] for r in C]).pvalue)
out['cross']['n_labels']=C[0]['n_labels']; out['cross']['features']=len(C[0]['shared_features']); out['cross']['zero_fpr']=ms([r['zero_shot']['fpr'] for r in C])
json.dump(out,open('paper/numbers_adaptation.json','w'),indent=1)
for r in out['lambda']: print(r['lam'],r['buffer'],r['n'],[round(x,1) for x in r['forg_f1']],round(r['new_before'][0],1))
print(out['buffer_test']); 
for c,v in out['stream'].items(): print(c,[round(v[k][0],1) for k in ['acc','f1','ret_acc','ret_f1','events','adapt_min']])
print('stream n',out['stream_n'],out['stream_p_replay_vs_naive_retF1'])
print({k:[round(x,2) for x in v['f1']]+[round(v['src'][0],1)] if isinstance(v,dict) and 'f1' in v else v for k,v in out['cross'].items()})

# ---- fair architecture comparison (Section 5.1, Table 5 last row) ----
# AML-IDS was trained with SMOTE, the deep baselines without it. abl_attention is the AML-IDS
# architecture trained without SMOTE on the same folds, so it isolates the effect of the architecture.
def _f1s(ds, m):
    runs = [d for d in ('results/indomain', 'results/indomain_rerun') if glob.glob(f'{d}/{ds}/{m}/fold*.json')]
    R = [[json.load(open(f)) for f in sorted(glob.glob(f'{d}/{ds}/{m}/fold*.json'))] for d in runs]
    return np.array([np.mean([cmf1(run[k]['confusion']) for run in R]) for k in range(5)])
fair = {}
for ds in ('cicids2017', 'unsw_nb15', 'nslkdd'):
    dsf = FOLDER.get(ds, ds)                       # UNSW-NB15: shuffled copy, as in Table 5
    a = _f1s(dsf, 'abl_attention'); s_ = _f1s(dsf, 'aml_ids')
    fair[ds] = {'aml_ids_no_smote': ms(a), 'smote_effect': float((s_ - a).mean())}
    for m in ('lstm', 'cnn_lstm_attn', 'mamba', 'transformer', 'cnn', 'rf', 'svm'):
        b = _f1s(dsf, m); p = float(ttest_rel(a, b).pvalue)
        fair[ds][m] = dict(baseline_f1=ms(b), diff=float((a - b).mean()), p=p, p_bonf=min(1.0, 7 * p))
# ---- UNSW-NB15 record-order check (Section 4.3): original file order vs. randomly shuffled copy ----
order = {}
for m in ('rf', 'abl_attention', 'lstm'):
    if glob.glob(f'results/indomain/unsw_shuffled/{m}/fold*.json'):
        order[m] = dict(ordered=ms(_f1s('unsw_nb15', m)), shuffled=ms(_f1s('unsw_shuffled', m)))
json.dump(dict(fair=fair, unsw_order_check=order), open('paper/numbers_fair_comparison.json', 'w'), indent=1)
print('SMOTE effect:', {d: round(v['smote_effect'], 1) for d, v in fair.items()})
print('UNSW order check:', {m: (round(v['ordered'][0], 1), round(v['shuffled'][0], 1)) for m, v in order.items()})
