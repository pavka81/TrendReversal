# src/model.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def train_model(
    X_train,
    y_train,
    param_grid: dict = None,
    cv_splits: int = 5
):
    """
    Train a RandomForest-based model within a pipeline (scaling + classifier).

    Parameters
    ----------
    X_train : pd.DataFrame or array-like
        Training features.
    y_train : pd.Series or array-like
        Training labels (boolean or binary).
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV. If None, uses default params.
    cv_splits : int
        Number of splits for TimeSeriesSplit cross-validation.

    Returns
    -------
    model : estimator
        Trained pipeline (either GridSearchCV or simple Pipeline).
    """


    """
    Train a RandomForest-based model within a pipeline (scaling + classifier).
    If param_grid is provided, does a GridSearchCV over it, otherwise uses
    the consensus defaults baked into the classifier.
    """
    # Base pipeline with consensus defaults
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=50,        # from consensus_params.json
            max_depth=5,            # cast float → int
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        ))
    ])

    # If someone passes a grid, still let them re−tune
    if param_grid:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=tscv,
            scoring='f1',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        return grid
    else:
        pipeline.fit(X_train, y_train)
        return pipeline