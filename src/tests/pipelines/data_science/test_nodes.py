from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, column
from hypothesis.extra.numpy import arrays
import pandas as pd
from kedro.config import ConfigLoader
from typing import List
import numpy as np
from sentiment_clsf.pipelines.data_science import nodes


@given(
    arrays(float,
           st.tuples(
               st.integers(min_value=2, max_value=100),
               st.integers(min_value=2400, max_value=3000))),
)
def test_split_data(features, labels):
    conf_paths = ["conf/base"]
    conf_loader = ConfigLoader(conf_paths)
    params = conf_loader.get("parameters*", "parameters*/**")
    try:
        assert len(nodes.split_data(features, labels, params)) == 6
        assert isinstance(nodes.split_data(df_agg, random_state)[1], pd.Series)
        assert isinstance(nodes.split_data(df_agg, random_state)[3], pd.Series)

        assert (len(split(df_agg, random_state)[0]) + len(split(df_agg, random_state)[1])) == len(df_agg)

        # если разбивка будет собственная, то использовать такую проверку
        #pd.testing.assert_frame_equal(expected_train_X, train_X)
        #pd.testing.assert_frame_equal(expected_test_X, test_X)

        # идекса из train нет в
        for i in split(df_agg, random_state)[1].index.values:
            assert i not in split(df_agg, random_state)[0].index
        # правильно np.allclose(split(df_agg, random_state)[1], split(df_agg, random_state)[0])
    except ValueError:
        # если данные не сплитятся или таргет одно значене
        True
