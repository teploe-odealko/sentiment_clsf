from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, column
import pandas as pd
from kedro.config import ConfigLoader
from typing import List
import numpy as np
from sentiment_clsf.pipelines.data_engineering import nodes

@given(
    data_frames(
        [
            column('text', dtype=object,
                   elements=st.text()),
            column('label', dtype=int,
                   elements=st.integers(min_value=0, max_value=1)),
        ]
    ),
)
def test_generate_features(df):
    conf_paths = ["conf/base"]
    conf_loader = ConfigLoader(conf_paths)
    params = conf_loader.get("parameters*", "parameters*/**")

    # smoke
    assert callable(nodes.generate_features) is True
    # type
    res = nodes.generate_features(df, params)
    assert isinstance(res, List)
    assert isinstance(res[0], np.ndarray)
    assert isinstance(res[1], np.ndarray)
    # unit
    features = res[0]
    labels = res[1]
    assert labels.shape[0] == features.shape[0]
    if features.shape[0] != 0:
        assert labels.shape[0] == features.shape[0] == len(df)
        assert features.shape[1] == params['k_best_features']


@settings(deadline=1200)
@given(
    data_frames(
        [
            column('text', dtype=object,
                   elements=st.text()),
            column('label', dtype=int,
                   elements=st.integers(min_value=0, max_value=1)),
        ]
    ),
)
def test_preprocess_reviews(df):
    conf_paths = ["conf/base"]
    conf_loader = ConfigLoader(conf_paths)
    params = conf_loader.get("parameters*", "parameters*/**")

    # smoke
    assert callable(nodes.preprocess_reviews) is True
    # type
    res = nodes.preprocess_reviews(df)
    assert isinstance(res, pd.DataFrame)

    # unit
    assert len(res) == len(df)
    assert list(df.columns) == list(res.columns)
