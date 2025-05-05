# src/hyperparams.py

param_grid = {
    'clf__n_estimators':    [50, 100, 200],
    'clf__max_depth':       [None, 5, 10, 20],
    'clf__min_samples_leaf':[1, 5, 10],
    'clf__max_features':    ['sqrt', 'log2', 0.5],
}

