# Exercise 5

This chapter covers classification problems: binary/multiclass on MNIST and Bank Customer Churn prediction.

## Local runner

Install chapter dependencies:

```bash
python -m pip install -r exercise_5/requirements.txt
```

Run the chapter-local launcher:

```bash
python exercise_5/main.py
```

Run one or more specific steps:

```bash
python exercise_5/main.py --exercise 5_1
python exercise_5/main.py --exercise 5_2
python exercise_5/main.py --exercise all
```

## Steps

- `5_1`: MNIST digit classification — binary is-5 detector (confusion matrix, precision/recall/F1, ROC/AUC) and multiclass 10-digit SGD classifier. Plots saved to `exercise_5/step_1/images/`.
- `5_2`: Bank Customer Churn — baseline Random Forest vs GridSearchCV with feature engineering (LabelBinarizer, RobustScaler, VarianceThreshold) across RandomForest, GradientBoosting, AdaBoost, and KNeighbors. Evaluation at 80% TPR threshold. Plots saved to `exercise_5/step_2/images/`.

## Notes

- Exercise 5.1 downloads the MNIST dataset (~50 MB) on first run via `sklearn.datasets.fetch_openml`. It is cached automatically.
- Exercise 5.2 runs GridSearchCV across 4 classifier families (several minutes on first run). The result is cached in `step_2/grid_search_cache.pkl` — delete it to retrain.
- The professor's original scripts are kept in `step_2/reference/` for comparison.

Use the root `main.py` only if you want the top-level launcher that dispatches to all chapters.
