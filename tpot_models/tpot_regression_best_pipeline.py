import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -0.21938250926107025
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=1.0, fit_intercept=False, l1_ratio=1.0, learning_rate="invscaling", loss="epsilon_insensitive", penalty="elasticnet", power_t=10.0)),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    XGBRegressor(learning_rate=0.1, max_depth=8, min_child_weight=3, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.6500000000000001, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
