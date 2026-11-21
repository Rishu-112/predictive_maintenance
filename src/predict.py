"""Predict RUL at the latest observed cycle for each engine from raw trajectory file."""
import argparse
import joblib
import numpy as np
import pandas as pd
from .data import read_trajectory, features


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,help='NASA-style 26-column trajectory .txt')
    parser.add_argument('--model',default='artifacts/rul_model.joblib')
    parser.add_argument('--output',default='reports/new_predictions.csv')
    args=parser.parse_args()
    bundle=joblib.load(args.model)
    frame,_=features(read_trajectory(args.input))
    last=frame.groupby('engine_id').tail(1)
    estimates=np.maximum(0,bundle['pipeline'].predict(last[bundle['features']]))
    result=pd.DataFrame({'engine_id':last.engine_id.values,'last_observed_cycle':last.cycle.values,'predicted_rul':estimates})
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(result.to_string(index=False))
    print('Saved',args.output)

if __name__=='__main__':main()
