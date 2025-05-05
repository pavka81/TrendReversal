# src/analysis.py

import pandas as pd
import matplotlib.pyplot as plt


def show_feature_importances(model, feature_names):
    """
    Extract and display RandomForest feature importances from a trained Pipeline or GridSearchCV.

    Parameters
    ----------
    model : estimator
        A fitted Pipeline or GridSearchCV containing a 'clf' step with RandomForestClassifier.
    feature_names : list of str
        List of feature names corresponding to the order used in the model.

    Returns
    -------
    pd.Series
        Sorted feature importances (descending).
    """
    # Unwrap GridSearchCV
    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_

    # Get the RandomForest classifier
    rf = model.named_steps.get('clf')
    try:
        importances = rf.feature_importances_
    except AttributeError:
        raise ValueError("Model does not have 'feature_importances_' attribute.")

    # Build a pandas Series for easy viewing
    imp_series = pd.Series(importances, index=feature_names)
    imp_series = imp_series.sort_values(ascending=False)

    # Print importances
    print("\nFeature importances:")
    print(imp_series.to_string())

    return imp_series


def plot_feature_importances(importances, top_n=None):
    """
    Plot a horizontal bar chart of feature importances.

    Parameters
    ----------
    importances : pd.Series
        Feature importances, indexed by feature name.
    top_n : int, optional
        If provided, plot only the top_n features.
    """
    if top_n is not None:
        importances = importances.head(top_n)
    # Sort for horizontal bar plot
    importances.sort_values().plot.barh()
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
