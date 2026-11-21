"""Train with engine-disjoint validation; evaluate official held-out engines once."""
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .data import features, load_data


def score(y,p):
    return {'MAE':round(float(mean_absolute_error(y,p)),3),
            'RMSE':round(float(np.sqrt(mean_squared_error(y,p))),3),
            'R2':round(float(r2_score(y,p)),3)}


def last_per_engine(df):
    return df.groupby('engine_id',sort=False).tail(1).index


def models():
    return {
      'median_baseline': Pipeline([('imputer',SimpleImputer(strategy='median')),('model',DummyRegressor(strategy='median'))]),
      'random_forest': Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=120,max_depth=14,min_samples_leaf=3,n_jobs=-1,random_state=42))]),
      'hist_gradient_boosting': Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingRegressor(max_iter=180,learning_rate=.07,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))])
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data/raw')
    ap.add_argument('--demo',action='store_true',help='Train on synthetic demo data, not NASA')
    ap.add_argument('--artifacts',default='artifacts')
    ap.add_argument('--reports',default='reports')
    args=ap.parse_args()
    out=Path(args.artifacts); report=Path(args.reports)
    out.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    train,test=load_data(args.data_dir,args.demo)
    tr,cols=features(train); te,_=features(test)
    splitter=GroupShuffleSplit(test_size=.25,n_splits=1,random_state=42)
    a,b=next(splitter.split(tr[cols],tr.rul,groups=tr.engine_id))
    fit,valid=tr.iloc[a],tr.iloc[b]
    # Choose model using one prediction per validation engine, to match official test evaluation.
    # Mimic official test truncation: one random pre-failure checkpoint per validation engine.
    rng=np.random.default_rng(42)
    checkpoint_indices=[]
    for _, engine in valid.groupby('engine_id'):
        life=int(engine.cycle.max())
        stop=int(rng.integers(max(1,int(life*.5)),max(2,int(life*.9))))
        checkpoint_indices.append(engine.index[engine.cycle.to_numpy()<=stop][-1])
    idx=pd.Index(checkpoint_indices)
    results={}; fitted={}
    for name,est in models().items():
        est.fit(fit[cols],fit.rul)
        pred=est.predict(valid[cols])
        pred_final=pd.Series(pred,index=valid.index).loc[idx]
        results[name]={'validation_all_cycles':score(valid.rul,pred),
                       'validation_truncated_checkpoint':score(valid.loc[idx,'rul'],pred_final)}
        fitted[name]=est
        print(name,results[name])
    winner=min(results,key=lambda n:results[n]['validation_truncated_checkpoint']['RMSE'])
    # Refit on all training engines only after choosing winner. Official test untouched until now.
    best=models()[winner]
    best.fit(tr[cols],tr.rul)
    test_last=te.loc[last_per_engine(te)].copy()
    p=best.predict(test_last[cols])
    final=score(test_last.rul,p)
    joblib.dump({'pipeline':best,'features':cols,'demo':args.demo,'model':winner},out/'rul_model.joblib')
    predictions=pd.DataFrame({'engine_id':test_last.engine_id.to_numpy(),
                              'observed_cycle':test_last.cycle.to_numpy(),
                              'actual_rul':test_last.rul.to_numpy(),
                              'predicted_rul':np.maximum(0,p)})
    predictions.to_csv(report/'test_predictions.csv',index=False)
    summary={'dataset':'SYNTHETIC DEMO (NOT NASA)' if args.demo else 'NASA C-MAPSS FD001',
             'train_engines':int(tr.engine_id.nunique()),'test_engines':int(te.engine_id.nunique()),
             'selection_metric':'RMSE on a randomly truncated pre-failure checkpoint per validation engine',
             'validation':results,'chosen_model':winner,'official_test_last_cycle':final,
             'caveat':'Simulation data; not verified for deployment or maintenance decisions.'}
    (report/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(test_last.rul,p,alpha=.65)
    limit=max(float(np.max(test_last.rul)),float(np.max(p)),1)
    ax.plot([0,limit],[0,limit],'--',color='gray')
    ax.set(xlabel='Actual remaining cycles',ylabel='Predicted remaining cycles',title='Held-out engine predictions')
    fig.tight_layout();fig.savefig(report/'actual_vs_predicted.png',dpi=150);plt.close(fig)
    print('\nSelected model:',winner)
    print('Official test set (last cycle per engine):',final)
    print('Outputs:',out/'rul_model.joblib',report/'metrics.json')
    if args.demo: print('WARNING: Synthetic demonstration results are NOT NASA benchmark results.')

if __name__=='__main__':
    main()
