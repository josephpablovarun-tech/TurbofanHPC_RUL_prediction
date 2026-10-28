# model_save

Tuned models from `model/03_model_tuning.ipynb`, each trained on the 70 training engines (seed 42, 70:30 split).

| File | Model |
|---|---|
| `logistic_regression_bands.joblib` | Logistic Regression on 10-cycle bands |
| `random_forest.joblib` | Random Forest, 300 trees |
| `xgboost.joblib` | XGBoost, 400 trees, max depth 7, square-root target |

Each file is a dictionary with the fitted `model`, its `features` (column names, in order), the `target` form, the tuned `settings`, `train_engine_ids` and the library `versions` it was saved with.

```python
import sys, joblib
sys.path.insert(0, '<project root>')            # so 'src' can be imported
from src.rul_models import predict_rul         # also needed to unpickle the logistic model
bundle = joblib.load('model_save/xgboost.joblib')
rul = predict_rul(bundle, feature_frame)       # feature_frame must hold the columns in bundle['features']
```

The feature columns are built by `build_features()` in section 2 of the notebook (causal rolling features). Load with the same scikit-learn and XGBoost versions listed in each bundle, or expect version warnings.
