import numpy as np
from src.data import demo_data,features,add_train_rul
from src.train import last_per_engine


def test_rul_decreases_with_cycles():
    train,_=demo_data(n_train=4,n_test=2)
    for _,g in train.groupby('engine_id'):
        assert g.rul.iloc[-1]==0
        assert np.all(np.diff(g.rul.to_numpy())==-1)


def test_features_are_causal():
    train,_=demo_data(n_train=2,n_test=2)
    full,cols=features(train)
    engine=full[full.engine_id==1]
    truncated=train[(train.engine_id==1)&(train.cycle<=15)]
    part,_=features(truncated)
    np.testing.assert_allclose(engine[cols].iloc[:15].to_numpy(),part[cols].to_numpy(),equal_nan=True)


def test_endpoints_one_per_engine():
    train,_=demo_data(n_train=5,n_test=2)
    assert len(last_per_engine(train))==5
