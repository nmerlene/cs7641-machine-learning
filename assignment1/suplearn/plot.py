# -*- coding: utf-8 -*-
"""Supervised Learning - Plotting routines

References:
    * https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    * https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    * https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine
    * https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from suplearn import config

# Set font sizes
plt.rc("font", size=config.FONTSIZE_SMALL)
plt.rc("axes", titlesize=config.FONTSIZE_BIG)
plt.rc("axes", labelsize=config.FONTSIZE_MEDIUM)
plt.rc("xtick", labelsize=config.FONTSIZE_SMALL)
plt.rc("ytick", labelsize=config.FONTSIZE_SMALL)
plt.rc("legend", fontsize=config.FONTSIZE_MEDIUM)
plt.rc("figure", titlesize=config.FONTSIZE_BIG)


def plot_confusion_matrix(data, ax=None):
    """Plots confusion matrix"""
    confusion_matrix_display = metrics.plot_confusion_matrix(
        data["estimator"], data["x_test"], data["y_test"], ax=ax
    )
    return confusion_matrix_display.ax_.get_figure(), confusion_matrix_display.ax_


def plot_learning_curve(data, ax=None):
    """Plot learning curve"""
    # Cast data to numpy arrays
    data = {key: np.array(value) for key, value in data.items() if isinstance(value, list)}
    # Generate figure
    if not ax:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
    else:
        fig = ax.get_figure()
    # Plot learning curve
    ax.grid()
    ax.fill_between(
        data["train_sizes"],
        data["train_scores_mean"] - data["train_scores_std"],
        data["train_scores_mean"] + data["train_scores_std"],
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        data["train_sizes"],
        data["test_scores_mean"] - data["test_scores_std"],
        data["test_scores_mean"] + data["test_scores_std"],
        alpha=0.1,
        color="g",
    )
    ax.plot(
        data["train_sizes"], data["train_scores_mean"], "o-", color="r", label="Training score"
    )
    ax.plot(
        data["train_sizes"],
        data["test_scores_mean"],
        "o-",
        color="g",
        label="Cross-validation score",
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    if config.SCORE_FIX_YLIM:
        ax.set_ylim(config.SCORE_YLIM)
    ax.legend(loc=config.LEGEND_LOC)
    return fig, ax


def plot_scalability(data, ax=None):
    """Plot scalability of model"""
    # Cast data to numpy arrays
    data = {key: np.array(value) for key, value in data.items() if isinstance(value, list)}
    # Generate figure
    if not ax:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
    else:
        fig = ax.get_figure()
    # Plot n_samples vs fit_times
    ax.grid()
    ax.plot(data["train_sizes"], data["fit_times_mean"], "o-")
    ax.fill_between(
        data["train_sizes"],
        data["fit_times_mean"] - data["fit_times_std"],
        data["fit_times_mean"] + data["fit_times_std"],
        alpha=0.1,
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Fit Times")
    return fig, ax


def plot_performance(data, ax=None):
    """Plot performance of model"""
    # Cast data to numpy arrays
    data = {key: np.array(value) for key, value in data.items() if isinstance(value, list)}
    # Generate figure
    if not ax:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
    else:
        fig = ax.get_figure()
    # Plot fit_time vs score
    ax.grid()
    ax.plot(data["fit_times_mean"], data["test_scores_mean"], "o-")
    ax.fill_between(
        data["fit_times_mean"],
        data["test_scores_mean"] - data["test_scores_std"],
        data["test_scores_mean"] + data["test_scores_std"],
        alpha=0.1,
    )
    ax.set_xlabel("Fit Times")
    ax.set_ylabel("Score")
    if config.SCORE_FIX_YLIM:
        ax.set_ylim(config.SCORE_YLIM)
    return fig, ax


def plot_validation_curve(data, ax=None):
    """Plot validation curve"""
    # Cast data to numpy arrays
    data = {
        key: np.array(value) if isinstance(value, list) else value
        for key, value in data.items()
    }
    # Massage param range to strings
    data["param_range"] = [str(item) for item in data["param_range"]]
    # Generate figure
    if not ax:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
    else:
        fig = ax.get_figure()
    # Plot validation curve
    ax.grid()
    ax.set_xlabel(data["param_name"])
    ax.set_ylabel("Score")
    if config.SCORE_FIX_YLIM:
        ax.set_ylim(config.SCORE_YLIM)
    lw = 2
    ax.plot(
        data["param_range"],
        data["train_scores_mean"],
        label="Training score",
        color="darkorange",
        lw=lw,
    )
    ax.fill_between(
        data["param_range"],
        data["train_scores_mean"] - data["train_scores_std"],
        data["train_scores_mean"] + data["train_scores_std"],
        alpha=0.2,
        color="darkorange",
        lw=lw,
    )
    ax.plot(
        data["param_range"],
        data["test_scores_mean"],
        label="Cross-validation score",
        color="navy",
        lw=lw,
    )
    ax.fill_between(
        data["param_range"],
        data["test_scores_mean"] - data["test_scores_std"],
        data["test_scores_mean"] + data["test_scores_std"],
        alpha=0.2,
        color="navy",
        lw=lw,
    )
    ax.legend(loc=config.LEGEND_LOC)
    return fig, ax


def plot_loss_curve(data, ax=None):
    """Plot loss curve"""
    # Generate figure
    if not ax:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
    else:
        fig = ax.get_figure()
    ax.plot(data["estimator"].loss_curve_)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc=config.LEGEND_LOC)
    return fig
