#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supervised Learning - Main script to run pipeline"""
import argparse
import datetime
import glob
import json
import logging
import os
import shutil

from suplearn import config
from suplearn import data
from suplearn import learners

# Logger
logging.basicConfig(**config.LOGGING_CONFIG)
LOGGER = logging.getLogger(__name__)


def init_run(root_run_dir, name=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")):
    """Initialize run"""
    # Create run directory
    run_dir = os.path.abspath(
        os.path.expanduser(os.path.expandvars(os.path.join(root_run_dir, name)))
    )
    os.makedirs(run_dir, exist_ok=True)
    # Write hyperparameters config to JSON in run directory
    with open(os.path.join(run_dir, "config.json"), "w") as open_file:
        open_file.write(json.dumps(config.PARAMS, indent=4))
    return run_dir


def get_inputs():
    """
    Extract inputs from command line

    Returns:
        args (argparse.NameSpace): Argparse namespace object
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-d", "--dataset", help="Dataset", required=True)
    parser.add_argument("-l", "--learner", help="Learner", required=True)
    parser.add_argument(
        "-g",
        "--grid-search",
        help="Run a grid search before running the learner",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--bias-variance",
        help="Compute bias-variance decomposition",
        action="store_true",
    )
    parser.add_argument(
        "--defaults",
        help='Run learner with default params ("--grid-search" will be ignored if this is used)',
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--run-name",
        help="Name for run directory",
        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "-r",
        "--root-run-dir",
        help="Root run directory",
        default=None,
    )
    parser.add_argument(
        "--copy-figures-path",
        help="Copy figures to the input directory name",
        default=None,
    )
    parser.add_argument(
        "--copy-figures-suffix",
        help='Suffix to add to figures if "--copy-figures-path" is used',
        default=None,
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    # Extract inputs
    args = get_inputs()
    # Massage learner name
    args.learner = f'{args.learner.replace("Learner", "")}Learner'
    # Set root run directory
    if not args.root_run_dir:
        args.root_run_dir = os.path.join(config.RUNS_PATH, args.learner, args.dataset)
    # Initialize run
    run_dir = init_run(args.root_run_dir, args.run_name)
    # Run dataset
    LOGGER.info(f"Run directory: {run_dir}")
    LOGGER.info(f"Dataset: {args.dataset}")
    LOGGER.info(f"Learner: {args.learner}")
    # Read dataset
    dataset = data.Data(
        args.dataset,
        os.path.join(config.DATA_PATH, config.PARAMS[args.dataset]["filename"]),
        config.PARAMS[args.dataset],
    )
    # Pre-process dataset
    dataset.preprocess()
    # Construct learner (will be initialized with params that are held constant)
    learner = getattr(learners, args.learner)(dataset)
    # Run analysis
    learner.do_all(run_dir, args.defaults, args.grid_search, args.bias_variance)
    # Optionally copy figures to another directory
    if args.copy_figures_path:
        filenames = glob.glob(os.path.join(run_dir, "*.png"))
        for src in filenames:
            # Split filename
            path, ext = os.path.splitext(os.path.basename(src))
            # Optionally add suffix to filename
            if args.copy_figures_suffix:
                dst = os.path.join(
                    args.copy_figures_path, f"{path}_{args.copy_figures_suffix}{ext}"
                )
            else:
                dst = os.path.join(args.copy_figures_path, f"{path}{ext}")
            # Copy figure
            LOGGER.info(f"Copying {src} to {dst}")
            shutil.copyfile(src, dst)
    LOGGER.info("Done!\n")


if __name__ == "__main__":
    main()
