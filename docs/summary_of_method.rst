.. _summary-of-modules-and-method:

Summary of modules and method
========
The method follows that presented in `Logan and Fotopoulou (2020) <https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract>`_ and
makes use of `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/>`_ as the unsupervised clusterer.
It is recommended to read that paper to fully understand the method used here.

In short, our method takes colours as input, and
selects the most important colours to use as attributes by running a Random Forest (RF)
to rank the importances of the features. Various feature combinations are then reduced
to lower dimensions via Principal Component Analysis (PCA), and fed to HDBSCAN, which outputs cluster labels for each
data point. The optimal setup for HDBSCAN (i.e. which input attributes, and what hyperparameter setup
to use for the algorithm HDBSCAN itself) is found via a gridsearch, and the final optimal
setup is then selected.

There are flowcharts in :ref:`binary` and :ref:`consolidation` that show an overview
of the optimization procedure used for the training of HDBSCAN. In order to achieve
good performance, we found that it is important to 1) select informative features
using RF to rank feature importances and 2) reduce the
dimensionality in order to remove correlations from the attributes and subsequently
present HDBSCAN with a more manageable number of dimensions where it performs best.
While a RF classifier can be presented with a large number of correlated features,
HDBSCAN would be impacted both in terms of classification quality and computation time.

We therefore start from all colour combinations available in our dataset,
and use a RF classifier and the spectroscopic labels to identify the relative rank of all
input features, keeping the most informative. Using the PCA implementation of scikit learn we further reduce these features
to a lower number of dimensions. Finally, we perform a gridsearch on the ``min_cluster_size``
to identify the best hyperparameter setup for each of our HDBSCAN classifiers.

Following this procedure for each of the classes (star, galaxy, QSO), we construct
three binary classifiers (see :ref:`binary`). Having classified our
sample three times, we apply the consolidation step (see :ref:`consolidation`),
leading to the final object classification including star, galaxies, QSO and outliers.

In addition, after using the method described above to train three binary classifiers,
the trained HDBSCAN clusterers can then be used to predict the cluster class
of new data points, which can then be fed into the consolidation step to give
output predicted labels for a completely new dataset (see :ref:`predict`).

A couple of uses for this method are: train the model on a dataset with known labels,
and having obtained predicted labels for each datapoint in this dataset, locate the datapoints whose labels from the clustering
method disagree with the initial known labels. These points can then be investigated further to
correct potentially mislabelled sources. An additional use is training on
a known dataset, and then applying the trained classifier on a new dataset to predict labels for
previously unseen data with no existing labels.

We describe the implementation of this method in the code in the links to each module
on the left. The code consists of three main scripts, :ref:`binary`, :ref:`consolidation` and
:ref:`predict`, which are all called in :ref:`main`.
