'''
Created on Nov 1, 2023

@author: simon
'''
from experiments.metrics import PhaseHistoryMetric, CosMetric, MeanDeviationMetric
from experiments.experiment import Experiment, CutOffExperiment, CutOffDataExperiment
from experiments.catalog import model_catalog
from experiments.plots import (
    phase_error_plot, phase_history_bias_plot, phase_history_metric_plot, triangle_experiment_plot)