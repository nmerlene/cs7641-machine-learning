# -*- coding: utf-8 -*-
"""Supervised Learning - Data Reading and Pre-Processing

References:
    * https://stackoverflow.com/questions/49444262/normalize-data-before-or-after-split-of-training-and-testing-data
    * https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
"""
import logging
import os

import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing

from suplearn import config

# Logger
LOGGER = logging.getLogger(__name__)


class Data:
    """Class to represent data container"""

    def __init__(self, name, filename, metadata, **kwargs):
        """
        Constructor for Data

        Args:
            name (str): Dataset name
            filename (str): Path to CSV data
            metadata (dict): Metadata for dataset
            **kwargs (dict): Any number of keyword-arguments to pass to pd.read_csv()
        """
        self.name = name
        self.filename = os.path.abspath(os.path.expanduser(filename))
        self.metadata = metadata
        self.data = pd.read_csv(self.filename, **kwargs)
        self.test_size = 0.25
        self.x = pd.DataFrame()
        self.y = pd.Series()
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series()
        self.y_test = pd.Series()
        self.y_predict = pd.Series()

    def preprocess(self, **kwargs):
        """
        Pre-process data:
        0. Convert label to a binary classification for the response variable
        1. Split into X and Y based off of input label
        2. Split into training and testing data
        3. Normalize data
        4. Standardize data

        Args:
             **kwargs (dict): Any number of keyword-arguments to pass to model_selection.test_train_split()
        """
        # Split into X and Y
        self.x = self.data.dropna().drop(self.metadata["label"], 1)
        self.y = self.data.dropna()[self.metadata["label"]]
        # Optionally trim X-data to just input features
        if self.metadata.get("features"):
            self.x = self.x[self.metadata["features"]]
        # Optionally bin label (Y-data) values
        if self.metadata.get("bin_label"):
            self.y = pd.cut(self.y, bins=self.metadata["bins"])
            self.y = preprocessing.LabelEncoder().fit_transform(self.y)
        # Optionally encode features
        if self.metadata.get("encoding"):
            # Label encoding columns
            label_encoding_cols = [
                key for key, value in self.metadata["encoding"].items() if value == "Label"
            ]
            # OneHot encoding columns
            one_hot_encoding_cols = [
                key for key, value in self.metadata["encoding"].items() if value == "OneHot"
            ]
            # LabelEncoding
            for col in label_encoding_cols:
                self.x[col] = preprocessing.LabelEncoder().fit_transform(self.x[col])
            # OneHotEncoding
            if one_hot_encoding_cols:
                self.x = pd.get_dummies(self.x, columns=one_hot_encoding_cols)
        # Stratify by default
        if "stratify" not in kwargs:
            kwargs["stratify"] = self.y
        # Set random state so that results are reproducible
        if "random_state" not in kwargs:
            kwargs["random_state"] = config.RANDOM_SEED
        # Split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.x, self.y, test_size=self.test_size, **kwargs
        )
        # Normalize data
        normalizer = preprocessing.Normalizer()
        self.x_train = normalizer.fit_transform(self.x_train)
        self.x_test = normalizer.transform(self.x_test)
        # Standardize data
        standard_scaler = preprocessing.StandardScaler()
        self.x_train = standard_scaler.fit_transform(self.x_train)
        self.x_test = standard_scaler.transform(self.x_test)
