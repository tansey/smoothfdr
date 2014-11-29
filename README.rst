False Discovery Rate Smoothing (smoothfdr)
------------------------------------------

The `smoothfdr` package provides an empirical-Bayes method for exploiting spatial structure in large multiple-testing problems. FDR smoothing automatically finds spatially localized regions of significant test statistics. It then relaxes the threshold of statistical significance within these regions, and tightens it elsewhere, in a manner that controls the overall false-discovery rate at a given level. This results in increased power and cleaner spatial separation of signals from noise. It tends to detect patterns that are much more biologically plausible than those detected by existing FDR-controlling methods.

For a detailed description of how FDR smoothing works, see the [paper on arXiv](http://arxiv.org/abs/1411.6144).

Installation
============
To install the Python version:

```
pip install smoothfdr
```

You can then run the tool directly from the terminal by typing the `smoothfdr` command. If you want to integrate it into your code, you can simply `import smoothfdr`.

The R source and package will be on CRAN and available publicly soon.


Running an example
==================

There are lots of parameters that you can play with if you so choose, but one of the biggest benefit of FDR smoothing is that you don't have to worry about it in most cases.

To run a simple example, we can generate our own synthetic data:

```
smoothfdr --signals_file test.signals --generate_data --signal_dist_name alt1 --estimate_signal --solution_path --data_file test.data \
 --plot_data test_data.pdf --plot_signal test_signal.pdf --plot_true_signal --plot_path test_path.pdf --plot_results test_results.pdf \
 --verbose 1 \
 2d
```

This will run the algorithm on a synthetic dataset that is generated on-the-fly. The algorithm will auto-tune its parameters by following a solution path approach where it tries multiple values. The result will be several plots:

### Visualizations of the true priors and raw data

![Visualizations of the true priors and raw data](https://raw.githubusercontent.com/tansey/smoothfdr/master/data/test_data.png)

### Density plots of the true and estimated signal distributions

![Density plots of the true and estimated signal distributions](https://raw.githubusercontent.com/tansey/smoothfdr/master/data/test_signal.png)

### Solution path diagnostics

![Solution path diagnostics](https://raw.githubusercontent.com/tansey/smoothfdr/master/data/test_path.png)

### Resulting plateaus detected

![Resulting plateaus detected](https://raw.githubusercontent.com/tansey/smoothfdr/master/data/test_results.png)

For a detailed list of commands, just run `smoothfdr -h`.

Still to come
=============
There are a few things we are still working on debugging in the production release but are ready in our private repo:

- Non-rectangular data (i.e. `fmri` mode) is currently buggy. This will be fixed soon.

- Empirical null is currently only available in the R version and will be integrated in future editions of the Python version.
