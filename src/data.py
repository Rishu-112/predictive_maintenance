"""NASA C-MAPSS FD001 data loading and an explicitly synthetic local demo."""
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
SENSOR_IDS = [2,3,4,7,11,12,15,20,21]
SENSORS = [f'sensor_{i}' for i in SENSOR_IDS]


def read_trajectory(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f'Expected {len(COLUMNS)} columns in {path}, got {df.shape[1]}')
    df.columns = COLUMNS
    df[['engine_id','cycle']] = df[['engine_id','cycle']].astype(int)
    return df.sort_values(['engine_id','cycle']).reset_index(drop=True)


def add_train_rul(df):
    df = df.copy()
    df['rul'] = df.groupby('engine_id')['cycle'].transform('max') - df['cycle']
    return df


def add_official_test_rul(test, rul_path):
    df = test.copy()
    offsets = pd.read_csv(rul_path, sep=r'\s+', header=None).iloc[:, 0].to_numpy(dtype=int)
    engine_ids = sorted(df.engine_id.unique())
    if len(engine_ids) != len(offsets):
        raise ValueError('Number of test engines does not match RUL labels')
    offset_map = dict(zip(engine_ids, offsets))
    max_cycle = df.groupby('engine_id')['cycle'].transform('max')
    df['rul'] = (max_cycle - df['cycle']) + df['engine_id'].map(offset_map)
    return df


def features(df):
    """Causal features only: no future observations or end-of-life info used."""
    x = df.sort_values(['engine_id','cycle']).copy()
    for sensor in SENSORS:
        group = x.groupby('engine_id')[sensor]
        x[f'{sensor}_avg5'] = group.transform(lambda s: s.rolling(5, min_periods=1).mean())
        x[f'{sensor}_std5'] = group.transform(lambda s: s.rolling(5, min_periods=2).std().fillna(0))
    cols = ['cycle'] + [f'setting_{i}' for i in range(1,4)] + SENSORS
    cols += [f'{s}_{stat}' for s in SENSORS for stat in ('avg5','std5')]
    return x, cols


def demo_data(n_train=36, n_test=12, seed=42):
    """Artificial sensor trajectories for a quick smoke test, NOT NASA observations."""
    rng = np.random.default_rng(seed)
    def build(n, test=False):
        rows, offsets = [], []
        for eid in range(1,n+1):
            life = int(rng.integers(105, 205))
            observed = int(rng.integers(55,life-8)) if test else life
            offsets.append(life-observed)
            for cycle in range(1,observed+1):
                aging = cycle/life
                sensor = [100 + 3*i + ((-1)**i)*24*aging**2 + rng.normal(0, 1.2) for i in range(1,22)]
                rows.append([eid,cycle,0.0,0.0,100.0,*sensor])
        frame = pd.DataFrame(rows,columns=COLUMNS)
        return frame,pd.Series(offsets)
    train,_ = build(n_train)
    test,offsets = build(n_test, test=True)
    return add_train_rul(train), add_official_test_rul_from_offsets(test,offsets)


def add_official_test_rul_from_offsets(test, offsets):
    df=test.copy()
    ids=sorted(df.engine_id.unique())
    mapping=dict(zip(ids,offsets.to_numpy()))
    df['rul'] = (df.groupby('engine_id')['cycle'].transform('max')-df['cycle'])+df.engine_id.map(mapping)
    return df


def load_data(data_dir, demo=False):
    if demo:
        return demo_data()
    root=Path(data_dir)
    files=[root/'train_FD001.txt',root/'test_FD001.txt',root/'RUL_FD001.txt']
    missing=[str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError('Missing NASA FD001 files: '+', '.join(missing)+'\nSee README for the dataset download.')
    train=add_train_rul(read_trajectory(files[0]))
    test=add_official_test_rul(read_trajectory(files[1]), files[2])
    return train,test
