# -*- coding: utf-8 -*-
"""Supervised Learning - Config file"""
import logging
import os

# Logging config
LOGGING_CONFIG = dict(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data")
RUNS_PATH = os.path.join(ROOT_PATH, "runs")

# Random seed for reproducibility
RANDOM_SEED = 69

# Matplotlib constants
FIGSIZE = (9, 4)
FONTSIZE_SMALL = 7
FONTSIZE_MEDIUM = 9
FONTSIZE_BIG = 11
SCORE_YLIM = (0.55, 1.05)
SCORE_FIX_YLIM = False
LEGEND_LOC = "lower right"

# Hyperparameters for each dataset
PARAMS = {
    "housing": {
        "filename": "melb_data.csv",
        "label": "Price",
        "features": [
            "Rooms",
            "Type",
            "Method",
            "Date",
            "Distance",
            "Postcode",
            "Bedroom2",
            "Bathroom",
            "Car",
            "Landsize",
            "BuildingArea",
            "YearBuilt",
            "Lattitude",
            "Longtitude",
            "Regionname",
            "Propertycount",
        ],
        "bin_label": True,
        "bins": [50000, 725000, 1150000, 10000000],
        "encoding": {
            "Type": "OneHot",
            "Method": "OneHot",
            "Date": "Label",
            "Regionname": "OneHot",
        },
        "params": {
            "DTLearner": {
                "class_weight": {"nested": False, "vals": ["balanced"]},
                "criterion": {"nested": False, "vals": ["gini"]},
                "max_depth": {"nested": False, "vals": [9]},
                "max_leaf_nodes": {"nested": False, "vals": [91]},
                "splitter": {"nested": False, "vals": ["best"]},
            },
            "NNLearner": {
                "alpha": {"nested": False, "vals": [1.0]},
                "batch_size": {"nested": False, "vals": [200]},
                "hidden_layer_sizes": {"nested": False, "vals": [[30, 30]]},
                "learning_rate": {"nested": False, "vals": ["constant"]},
                "max_iter": {"nested": False, "vals": [200]},
            },
            "BoostingLearner": {
                "algorithm": {"nested": False, "vals": ["SAMME.R"]},
                "base_estimator": {
                    "nested": True,
                    "module": "sklearn.tree",
                    "class": "DecisionTreeClassifier",
                    "arg": "max_depth",
                    "vals": [2],
                },
                "learning_rate": {"nested": False, "vals": [0.7]},
                "n_estimators": {"nested": False, "vals": [125]},
            },
            "SVMLearner": {
                "C": {"nested": False, "vals": [14]},
                "kernel": {"nested": False, "vals": ["poly"]},
                "tol": {"nested": False, "vals": [1e-3]},
                "degree": {"nested": False, "vals": [3]},
            },
            "KNNLearner": {
                "metric": {"nested": False, "vals": ["manhattan"]},
                "n_neighbors": {"nested": False, "vals": [16]},
                "weights": {"nested": False, "vals": ["uniform"]},
            },
        },
    },
    "wine": {
        "filename": "winequality-red.csv",
        "label": "quality",
        "bin_label": True,
        "bins": [1, 5.5, 10],
        "params": {
            "DTLearner": {
                "class_weight": {"nested": False, "vals": [None]},
                "criterion": {"nested": False, "vals": ["gini"]},
                "max_depth": {"nested": False, "vals": [14]},
                "max_leaf_nodes": {"nested": False, "vals": [102]},
                "splitter": {"nested": False, "vals": ["best"]},
            },
            "NNLearner": {
                "alpha": {"nested": False, "vals": [1e-5]},
                "batch_size": {"nested": False, "vals": [200]},
                "hidden_layer_sizes": {"nested": False, "vals": [[25, 25]]},
                "learning_rate": {"nested": False, "vals": ["constant"]},
                "max_iter": {"nested": False, "vals": [200]},
            },
            "BoostingLearner": {
                "algorithm": {"nested": False, "vals": ["SAMME.R"]},
                "base_estimator": {
                    "nested": True,
                    "module": "sklearn.tree",
                    "class": "DecisionTreeClassifier",
                    "arg": "max_depth",
                    "vals": [1],
                },
                "learning_rate": {"nested": False, "vals": [1.0]},
                "n_estimators": {"nested": False, "vals": [60]},
            },
            "SVMLearner": {
                "C": {"nested": False, "vals": [38]},
                "kernel": {"nested": False, "vals": ["poly"]},
                "tol": {"nested": False, "vals": [1e-3]},
                "degree": {"nested": False, "vals": [3]},
            },
            "KNNLearner": {
                "metric": {"nested": False, "vals": ["minkowski"]},
                "n_neighbors": {"nested": False, "vals": [5]},
                "weights": {"nested": False, "vals": ["uniform"]},
            },
        },
    },
}
