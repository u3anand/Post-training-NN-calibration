"""
This library contains the tools to plot reliability diagrams and confidence histograms
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_reliability_diagram(ax, confidences, accuracies, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    width = bins[1] - bins[0]
    bin_corrects = []
    bin_scores = []

    for i in range(n_bins):
        bin_indices = (confidences >= bins[i]) & (confidences < bins[i+1])
        if np.any(bin_indices):
            bin_corrects.append(np.mean(accuracies[bin_indices]))
            bin_scores.append(np.mean(confidences[bin_indices]))
        else:
            bin_corrects.append(0)
            bin_scores.append(0)

    confs = ax.bar(bins[:-1], bin_corrects, width=width, color='blue', edgecolor='black')
    gaps = ax.bar(bins[:-1], [bs - bc for bs, bc in zip(bin_scores, bin_corrects)], bottom=bin_corrects, color='red', alpha=0.5, width=width, hatch='//', edgecolor='black')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Confidence')
    ax.set_title('Reliability Diagram')
    ax.set_ylim([0, 1])
    ax.legend([confs, gaps], ['Accuracy', 'Gap'], loc='best', fontsize='small')
    ax.grid(True)

def plot_confidence_histogram(ax, confidences, accuracies, n_bins=10):
    counts, bin_edges = np.histogram(confidences, bins=n_bins, range=(0, 1))
    bin_width = bin_edges[1] - bin_edges[0]
    percent_samples = (counts / counts.sum()) * 100
    ax.bar(bin_edges[:-1], percent_samples, width=bin_width, color='skyblue', edgecolor='black', alpha=0.7, label='Confidence')
    ax.axvline(np.mean(confidences), color='green', linestyle='dashed', linewidth=2, label='Avg Confidence')
    ax.axvline(np.mean(accuracies), color='red', linestyle='dashed', linewidth=2, label='Avg Accuracy')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Percent of Samples')
    ax.set_title('Confidence Histogram')
    ax.legend()
    ax.grid(True)
