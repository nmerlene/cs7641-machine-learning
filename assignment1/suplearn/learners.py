# -*- coding: utf-8 -*-
"""
Supervised Learning - Learners

Learners:
* Boosting Learner
* Decision Tree (DT) Learner
* k-Nearest Neighbors (KNN) Learner
* Neural Networks (NN) Learner
* Support Vector Machines (SVM) Learner
"""
import copy
import json
import logging
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
from mlxtend import evaluate
from sklearn import ensemble
from sklearn import exceptions
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree
from sklearn.utils import _testing

from suplearn import config
from suplearn import plot

# Logger
LOGGER = logging.getLogger(__name__)


class BaseLearner:
    """Base Learner"""

    def __init__(self, learner, dataset):
        """Constructor for BaseLearner"""
        # Class attributes
        self.name = self.__class__.__name__
        self.func = learner
        self.learner = None
        self.data = dataset
        self.grid_search_cv = None
        self.params = self.get_params()
        self.vary_params = self.get_vary_params()
        self.constant_params = self.get_constant_params()
        self.best_params = dict()
        self.timing = dict()
        self.metrics = list()
        self.num_cores = round(multiprocessing.cpu_count() * 0.75)
        # Initialize learner
        self.init_learner(**self.constant_params)
        LOGGER.info(f"Number of cores: {self.num_cores}")

    @property
    def learner_str(self):
        """Return learner as a string"""
        return str(self.learner).replace("\n", "").replace(" ", "").replace(",", ", ")

    @staticmethod
    def array_to_string(array, num_decimals=2):
        """Convert numpy array to a string"""
        return " ".join(map(str, np.array(array).round(num_decimals).tolist()))

    def get_params(self):
        """Returns test params"""
        grid_search_params = dict()
        for key, value in self.data.metadata["params"][self.name].items():
            if value["nested"]:
                grid_search_params[key] = list()
                for iter_val in value["vals"]:
                    module = sys.modules[value["module"]]
                    constructor = getattr(module, value["class"])
                    instance = constructor(**{value["arg"]: iter_val})
                    grid_search_params[key].append(instance)
            else:
                grid_search_params[key] = value["vals"]
        return grid_search_params

    def get_vary_params(self):
        """Returns only the test params that are allowed to vary"""
        return {key: value for key, value in self.params.items() if len(value) > 1}

    def get_constant_params(self):
        """Returns only the test params that are held constant and NOT allowed to vary"""
        return {key: value[0] for key, value in self.params.items() if len(value) == 1}

    def get_metric(self, metric_type):
        """Returns a list of metric dictionaries based off of the metric type"""
        return list(filter(lambda x: x["type"] == metric_type, self.metrics))

    def init_learner(self, **kwargs):
        """Initializes/resets a learner with any number of keyword arguments"""
        try:
            self.learner = self.func(random_state=config.RANDOM_SEED, **kwargs)
        except TypeError:
            self.learner = self.func(**kwargs)
        if kwargs:
            LOGGER.info(f"Initialized learner: {self.learner_str}")
        else:
            LOGGER.info(f"Reset learner: {self.learner_str}")

    @_testing.ignore_warnings(category=exceptions.ConvergenceWarning)
    def grid_search(self):
        """Run grid search"""
        LOGGER.info(f"Running grid search")
        time_start = time.time()
        self.grid_search_cv = model_selection.GridSearchCV(
            self.learner,
            param_grid=self.params,
            n_jobs=self.num_cores,
        )
        self.grid_search_cv.fit(self.data.x_train, self.data.y_train)
        self.timing["grid_search"] = round(time.time() - time_start, 3)
        self.best_params = self.grid_search_cv.best_params_
        self.learner = self.grid_search_cv.best_estimator_
        LOGGER.info(f"Grid search selected the following learner: {self.learner_str}")

    def confusion_matrix(self):
        """Confusion matrix"""
        LOGGER.info("Generating confusion matrix")
        time_start = time.time()
        confusion_matrix = metrics.confusion_matrix(
            self.data.y_test, self.grid_search_cv.predict(self.data.x_test)
        )
        self.timing["confusion_matrix"] = round(time.time() - time_start, 3)
        self.metrics.append({"type": "confusion_matrix", "metric": confusion_matrix.tolist()})

    def learning_curve(self, **kwargs):
        """Learning curve"""
        LOGGER.info(f"Generating learning curve")
        time_start = time.time()
        (
            train_sizes,
            train_scores,
            test_scores,
            fit_times,
            score_times,
        ) = model_selection.learning_curve(
            self.learner,
            self.data.x_train,
            self.data.y_train,
            n_jobs=self.num_cores,
            return_times=True,
            **kwargs,
        )
        self.timing["learning_curve"] = round(time.time() - time_start, 3)
        self.metrics.append(
            {
                "type": "learning_curve",
                "metric": {
                    "train_sizes": train_sizes.tolist(),
                    "train_scores": train_scores.tolist(),
                    "test_scores": test_scores.tolist(),
                    "fit_times": fit_times.tolist(),
                    "score_times": score_times.tolist(),
                    "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
                    "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
                    "fit_times_mean": np.mean(fit_times, axis=1).tolist(),
                    "score_times_mean": np.mean(score_times, axis=1).tolist(),
                    "train_scores_std": np.std(train_scores, axis=1).tolist(),
                    "test_scores_std": np.std(test_scores, axis=1).tolist(),
                    "fit_times_std": np.std(fit_times, axis=1).tolist(),
                    "score_times_std": np.std(score_times, axis=1).tolist(),
                },
            }
        )

    def validation_curve(self, scoring="accuracy", **kwargs):
        """Validation curve"""
        if self.vary_params:
            LOGGER.info(f"Generating validation curve(s)")
        for param_name, param_range in self.vary_params.items():
            train_scores, test_scores = model_selection.validation_curve(
                self.learner,
                self.data.x_train,
                self.data.y_train,
                param_name=param_name,
                param_range=param_range,
                scoring=scoring,
                n_jobs=self.num_cores,
                **kwargs,
            )
            self.metrics.append(
                {
                    "type": "validation_curve",
                    "metric": {
                        "param_name": param_name,
                        "param_range": param_range,
                        "train_scores": train_scores.tolist(),
                        "test_scores": test_scores.tolist(),
                        "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
                        "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
                        "train_scores_std": np.std(train_scores, axis=1).tolist(),
                        "test_scores_std": np.std(test_scores, axis=1).tolist(),
                    },
                }
            )

    def cross_val_score(self):
        """Cross validation score"""
        LOGGER.info(f"Generating cross validation score")
        time_start = time.time()
        cross_val_score = model_selection.cross_val_score(
            self.learner, self.data.x_train, self.data.y_train, n_jobs=self.num_cores
        )
        self.timing["cross_val_score"] = round(time.time() - time_start, 3)
        self.metrics.append({"type": "cross_val_score", "metric": cross_val_score.tolist()})

    def bias_variance(self):
        """Bias variance"""
        LOGGER.info(f"Generating bias-variance decomposition")
        time_start = time.time()
        avg_expected_loss, avg_bias, avg_var = evaluate.bias_variance_decomp(
            self.learner,
            np.array(self.data.x_train),
            np.array(self.data.y_train),
            np.array(self.data.x_test),
            np.array(self.data.y_test),
            loss="mse",
            random_seed=config.RANDOM_SEED,
        )
        self.timing["bias_variance"] = round(time.time() - time_start, 3)
        self.metrics.append(
            {
                "type": "bias_variance",
                "metric": {
                    "avg_expected_loss": avg_expected_loss,
                    "avg_bias": avg_bias,
                    "avg_var": avg_var,
                },
            }
        )

    def fit(self):
        """Fit"""
        LOGGER.info(f"Running fit")
        time_start = time.time()
        self.learner.fit(self.data.x_train, self.data.y_train)
        self.timing["fit"] = round(time.time() - time_start, 3)

    def predict(self):
        """Predict"""
        LOGGER.info(f"Running predict")
        time_start = time.time()
        self.data.y_predict = self.learner.predict(self.data.x_test)
        self.timing["predict"] = round(time.time() - time_start, 3)

    def accuracy(self):
        """Accuracy (LCA)"""
        time_start = time.time()
        accuracy = metrics.accuracy_score(self.data.y_test, self.data.y_predict)
        self.timing["accuracy"] = round(time.time() - time_start, 3)
        self.metrics.append({"type": "accuracy", "metric": accuracy})

    def f1(self):
        """F1 Score"""
        time_start = time.time()
        accuracy = metrics.f1_score(
            self.data.y_test,
            self.data.y_predict,
            average="weighted",
            labels=np.unique(self.data.y_predict),
        )
        self.timing["f1"] = round(time.time() - time_start, 3)
        self.metrics.append({"type": "f1", "metric": accuracy})

    def classification_report(self):
        """Classification Report"""
        self.metrics.append(
            {
                "type": "classification_report",
                "metric": metrics.classification_report(
                    self.data.y_test, self.data.y_predict, output_dict=True, zero_division=1
                ),
            }
        )

    def to_dict(self, for_json=False):
        """Convert self to dictionary"""
        # Create dict
        output_dict = {
            "name": self.name,
            "learner": f"{self.func.__module__}.{self.func.__name__}",
            "filename": self.data.filename,
            "params": copy.deepcopy(self.params),
            "vary_params": copy.deepcopy(self.vary_params),
            "constant_params": copy.deepcopy(self.constant_params),
            "best_params": copy.deepcopy(self.best_params),
            "learner_params": self.learner.get_params(),
            "timing": self.timing,
            "metrics": self.metrics,
        }
        # Massage for JSON serialization if need be
        if for_json:
            # Check if JSON serializable as is
            try:
                json.dumps(output_dict)
            # If auto-generation doesn't work, convert any stubborn fields
            # to dicts/strings manually
            except TypeError:
                for key, value in output_dict.items():
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            try:
                                json.dumps(nested_value)
                            except TypeError:
                                if hasattr(nested_value, "__dict__"):
                                    output_dict[key][nested_key] = nested_value.__dict__
                                elif hasattr(nested_value, "__len__") and hasattr(
                                    nested_value[0], "__dict__"
                                ):
                                    output_dict[key][nested_key] = [
                                        iter_value.__dict__ for iter_value in nested_value
                                    ]
                                else:
                                    output_dict[key][nested_key] = str(nested_value)
                    else:
                        try:
                            json.dumps(value)
                        except TypeError:
                            output_dict[key] = str(value)
        return output_dict

    def to_json(self, filename):
        """Write to JSON file

        Args:
            filename (str): Path to JSON file to write
        """
        # Write to file
        with open(filename, "w") as open_file:
            open_file.write(json.dumps(self.to_dict(for_json=True), indent=4))

    def plot(self, dirname):
        """Generate plots

        Args:
            dirname (str): Output directory to save figures to
        """
        # Plot confusion matrix
        metric = self.get_metric("confusion_matrix")
        if metric:
            LOGGER.info("Plotting confusion matrix")
            fig, ax = plot.plot_confusion_matrix(
                {
                    "estimator": self.learner,
                    "x_test": self.data.x_test,
                    "y_test": self.data.y_test,
                }
            )
            ax.set_title(f"{self.data.name.title()}: Confusion Matrix ({self.name})")
            fig.savefig(
                os.path.join(dirname, f"{self.data.name}_{self.name}_confusion_matrix.png")
            )
        # Plot learning curve, scalability, and performance
        metric = self.get_metric("learning_curve")
        if metric:
            for plot_type in ("learning_curve", "scalability", "performance"):
                LOGGER.info(f"Plotting {plot_type}")
                fig, ax = getattr(plot, f"plot_{plot_type}")(metric[0]["metric"])
                ax.set_title(
                    f'{self.data.name.title()}: {plot_type.replace("_", " ").title()} ({self.name})'
                )
                fig.savefig(
                    os.path.join(dirname, f"{self.data.name}_{self.name}_{plot_type}.png")
                )
        # Plot validation curve(s)
        metric = self.get_metric("validation_curve")
        if metric:
            for param in metric:
                LOGGER.info(f'Plotting validation curve for: "{param["metric"]["param_name"]}"')
                fig, ax = plot.plot_validation_curve(param["metric"])
                ax.set_title(f"{self.data.name.title()}: Validation Curve ({self.name})")
                fig.savefig(
                    os.path.join(
                        dirname,
                        f'{self.data.name}_{self.name}_validation_curve_{param["metric"]["param_name"]}.png',
                    )
                )

    def print_summary(self):
        """Print summary"""
        # Dataset
        LOGGER.info("Dataset Summary:")
        LOGGER.info(f"    Original Shape: {self.data.data.shape}")
        LOGGER.info(
            f"    Original # of Null/NaN Values: "
            f"{self.data.data.shape[0] - self.data.data.dropna().shape[0]}/{self.data.data.shape[0]}"
        )
        LOGGER.info(
            f"    Original % of Null/NaN Values: "
            f"{(self.data.data.shape[0] - self.data.data.dropna().shape[0]) / self.data.data.shape[0] * 100:.3f}%"
        )
        LOGGER.info(f"    X-Data Shape: {self.data.x.shape}")
        LOGGER.info(f"    Y-Data Shape: {self.data.y.shape}")
        # Classification
        LOGGER.info("Classification Summary:")
        LOGGER.info(
            "    Features ("
            f'{len(self.data.metadata.get("features", self.data.x.columns))}/{self.data.data.shape[1] - 1}'
            "):"
        )
        for feature in self.data.metadata.get("features", self.data.x.columns):
            LOGGER.info(f"        {feature}")
        LOGGER.info(f"    Labels ({len(pd.Series(self.data.y).unique())}):")
        for ind, (key, value) in enumerate(
            dict(pd.Series(self.data.y).value_counts().sort_index()).items()
        ):
            if "bins" in self.data.metadata:
                LOGGER.info(
                    f'        {self.data.metadata["bins"][ind]} - {self.data.metadata["bins"][ind + 1]} '
                    f"({key}): {value} ({value / len(self.data.y) * 100:.3f}%)"
                )
            else:
                LOGGER.info(f"        {key}: {value} ({value / len(self.data.y) * 100:.3f}%)")
        LOGGER.info(f"    Train/Test Split: ({1 - self.data.test_size}, {self.data.test_size})")
        LOGGER.info(f"    X-Data (Train) Shape: {self.data.x_train.shape}")
        LOGGER.info(f"    Y-Data (Train) Shape: {self.data.y_train.shape}")
        LOGGER.info(f"    X-Data (Test) Shape: {self.data.x_test.shape}")
        LOGGER.info(f"    Y-Data (Test) Shape: {self.data.y_test.shape}")
        # Analysis
        LOGGER.info("Analysis Summary:")
        LOGGER.info("    Learner:")
        LOGGER.info(f"        {self.learner_str}")
        LOGGER.info("    Timing:")
        for key, value in self.timing.items():
            LOGGER.info(f"        {key}: {value}s")
        LOGGER.info("    Learning Curve:")
        # Learning curve
        metric = self.get_metric("learning_curve")
        if metric:
            stat = metric[0]["metric"]
            LOGGER.info(
                f'        Train scores (Mean): {self.array_to_string(stat["train_scores_mean"])}'
            )
            LOGGER.info(
                f'        Test scores (Mean): {self.array_to_string(stat["test_scores_mean"])}'
            )
            diff = np.array(stat["train_scores_mean"]) - np.array(stat["test_scores_mean"])
            LOGGER.info(f"        Diff (Mean): {self.array_to_string(diff)}")
        # Validation curve(s)
        metric = self.get_metric("validation_curve")
        if metric:
            LOGGER.info("    Validation Curve:")
            for param in metric:
                stat = param["metric"]
                name = stat["param_name"]
                LOGGER.info(
                    f'        Train scores (Mean) for {name}: {self.array_to_string(stat["train_scores_mean"])}'
                )
                LOGGER.info(
                    f'        Test scores (Mean) for {name}: {self.array_to_string(stat["test_scores_mean"])}'
                )
                diff = np.array(stat["train_scores_mean"]) - np.array(stat["test_scores_mean"])
                LOGGER.info(f"        Diff (Mean) for {name}: {self.array_to_string(diff)}")
        # Bias-Variance
        metric = self.get_metric("bias_variance")
        if metric:
            LOGGER.info("    Bias-Variance Decomposition:")
            LOGGER.info(
                f'        Avg Expected Loss: {metric[0]["metric"]["avg_expected_loss"]:.3f}'
            )
            LOGGER.info(f'        Avg Bias: {metric[0]["metric"]["avg_bias"]:.3f}')
            LOGGER.info(f'        Avg Variance: {metric[0]["metric"]["avg_var"]:.3f}')
        # Cross validation score
        LOGGER.info("    Scores:")
        metric = self.get_metric("cross_val_score")
        if metric:
            LOGGER.info(f'        Cross val score (Mean): {np.mean(metric[0]["metric"]):.3f}')
        # Accuracy and F1
        for score in ("accuracy", "f1"):
            metric = self.get_metric(score)
            if metric:
                LOGGER.info(f'        {score.title()}: {metric[0]["metric"]:.3f}')
        # Classification report
        LOGGER.info("    Full Classification Report:")
        lines = metrics.classification_report(
            self.data.y_test, self.data.y_predict, zero_division=1
        )
        for line in lines.split("\n"):
            LOGGER.info(f"        {line.title()}")

    def do_all(self, run_dir, defaults=False, grid_search=False, bias_variance=False):
        """Do all"""
        # If running defaults, reset learner to default params
        if defaults:
            self.init_learner()
        # Or if we're doing a grid search, do it now
        # Note: Updates learner with the "best estimator" returned from the grid search
        elif grid_search:
            # Grid search
            self.grid_search()
            # Generate confusion matrix
            self.confusion_matrix()
        # Generate learning curve
        self.learning_curve()
        # Generate validation curve(s) for any params that are varied
        if not defaults:
            self.validation_curve()
        # Generate cross validation score
        self.cross_val_score()
        # Optionally compute bias variance (it is slow AF)
        if bias_variance:
            self.bias_variance()
        # Fit
        self.fit()
        # Predict
        self.predict()
        # Accuracy
        self.accuracy()
        # F1
        self.f1()
        # Classification report
        self.classification_report()
        # Write learner data to JSON
        self.to_json(os.path.join(run_dir, f"{self.name}.json"))
        # Run plotters
        self.plot(run_dir)
        # Print summary
        self.print_summary()


class BoostingLearner(BaseLearner):
    """Boosting Learner"""

    def __init__(self, dataset):
        super().__init__(ensemble.AdaBoostClassifier, dataset)


class DTLearner(BaseLearner):
    """Decision Tree Learner"""

    def __init__(self, dataset):
        super().__init__(tree.DecisionTreeClassifier, dataset)


class KNNLearner(BaseLearner):
    """k-Nearest Neighbors Learner"""

    def __init__(self, dataset):
        super().__init__(neighbors.KNeighborsClassifier, dataset)


class NNLearner(BaseLearner):
    """Neural Networks Learner"""

    def __init__(self, dataset):
        super().__init__(neural_network.MLPClassifier, dataset)


class SVMLearner(BaseLearner):
    """Support Vector Machines Learner"""

    def __init__(self, dataset):
        super().__init__(svm.SVC, dataset)
