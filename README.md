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

Running an example on an arbitrary graph
========================================

If your problem is not structured simply according to a 1d, 2d, or 3d grid, then you will want to run using the __graph__ type. This uses the `pygfl` package to solve the underlying graph-fused lasso problem. You'll need a CSV file containing the list of edges in the format:

```
0,1
1,2
2,10
5,15
3,0
...
```

where the first and second numbers correspond to the index of the node in the graph. Assuming this file is `test/edges.csv`, the first step is to generate the setup files for `pygfl`:

```
maketrails file --infile test/edges.csv --savet test/trails.csv
```

You can then run FDR smoothing using the `graph` setting, applied to some z-score file called `test/data.csv`:

```
smoothfdr --data_file test/data.csv \
--empirical_null --estimate_signal --solution_path --dual_solver graph --fdr_level 0.1 \
--save_weights test/weights.csv --save_posteriors test/posteriors.csv --save_signal test/signal.csv --save_plateaus test/plateaus.csv \
--plot_path test/plot_path.pdf --plot_signal test/plot_signal.pdf \
graph --trails test/trails.csv
```

The first line simply specifies the data file containing the vector z-scores. The second line specifies that the null and alternative (signal) distributions should be estimated from the data, that a solution path approach should be used to auto-tune the hyperparameters, and that we're using a false discovery rate of 10%. The third and fourth lines specify where to save the various types of output from the algorithm-- note that the weights (AKA priors) and posteriors are what you are most likely interested in here. Finally, the last line specifies we're using the graph-fused lasso solver and provides the setup file we generated previously.



































