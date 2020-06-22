# Unsupervised classification of star, galaxy, QSOs with HDBSCAN

The method implemented here is based on that presented in [Logan & Fotopoulou 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract) (hereafter LF2020),
which uses photometric data to classify objects into stars, galaxies and QSOs.
A couple of potential uses of this method include:
- training the classifier using a training set with known labels in [binary_classifier.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/binary_classifier.py)
and [classifier_consolidation.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/classifier_consolidation.py), and then running the
[predict.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/predict.py) script on a new catalogue with
unseen labels, to predict labels for a completely new dataset.
- given a training set with known labels, the [binary_classifier.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/binary_classifier.py)
and [classifier_consolidation.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/classifier_consolidation.py)
scripts will return a catalogue with predicted labels for this catalogue from [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) (using
the training labels just to judge its performance, but not dictate the
clustering itself - i.e. it is semi-supervised). These new predicted labels can be
compared to the original labels, and any disagreements may highlight previously
misclassified sources. This is one advantage of using unsupervised learning.

One difference in this code compared to [LF2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract) is that there is a reduced
attribute selection step (to select the input attributes to the hdbscan gridsearch). However, similar performance
(see second table in Example / Test Run below) can be reached using the method in the implementation here.
We note that here we only implement the method where colours are used as input, and do
not include any half light radius information in the implementation here.

## Getting started

### Installation

To download the code repository here, run the following in a terminal, where you want the code to be downloaded:

```
git clone https://github.com/crispinlogan/StarGalaxyQSOClassification.git 
```

### Requirements
Requirements are given in the [requirements.txt](requirements.txt) file.
We used [anaconda](https://docs.anaconda.com/anaconda/install/), specifically using Python3.7.

The requirements can either be installed via pip

```
pip install requirements.txt
```

or via conda (a new environment could be created first).

```
conda install -c conda-forge astropy=4.0.1.post1
conda install -c conda-forge hdbscan=0.8.26
conda install -c conda-forge seaborn==0.10.0
conda install -c conda-forge matplotlib=3.1.3
conda install -c conda-forge pandas=1.0.3
conda install -c conda-forge numpy=1.18.1
conda install -c conda-forge scikit-learn=0.23.0
```

### Running the code

Once the repo is cloned, you need to set the following variables in the `ConfigVars` class in the [config file](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/config.py):
- `base_path` - this should be set to where the repo is downloaded e.g. `/home/yourname/StarGalaxyQSOClassification/`.
It should be set in the [config file](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/config.py) in the `ConfigVars` class.
- `n_jobs` - the number of cores on which to run. Automatically set to 50% of available cores (with a minimum of 1)
in the `ConfigVars` class in the [config file](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/config.py).
It can also be passed as an input argument when instantiating
the `conf` object (e.g. calling `conf = ConfigVars(test = True, n_jobs=10)`), or set manually, after instantiating
the `conf` object (e.g. by calling `conf.n_jobs` = 10 after instantiating the `conf` object) in [main.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/main.py) or any of the
[binary_classifier.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/binary_classifier.py),
[classifier_consolidation.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/classifier_consolidation.py) or
[predict.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/predict.py) scripts.
- input catalogue - in the [config file](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/config.py), the attribute `catname` needs to be changed
to your training catalogue name, the `targetname` attribute should be set to the columm name in your catalogue
that has the 'true' labels for the objects, the `hclass_dict` attribute links the numeric values for the labels
in the catalogue with the object name, and the `data_file` attribute uses the `catname` attribute to be
the path to the catalogue itself. 
- prediction catalogue - in the [config file](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/config.py), the attribute `catname_predicted` needs to be changed
to your unseen catalogue that you want labels to be predicted for. It is used in the `data_file_predicted` variable
that is the path to this new catalogue.

The input catalogues (for training and prediction) need to be in the same format
as the [example catalogue for training](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/data/input/CPz.csv), taken from [LF2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract)
and [example catalogue for prediction](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/data/input/short_KiDSVW.csv),
which is a random sub-sample of 50,000 data points from the [KiDSVW catalogue](https://ui.adsabs.harvard.edu/abs/2019yCat..36330154L/abstract),
which in turn is described in [LF2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract).
Depending on the photometric bands available in your catalogue, the variable `photo_band_list`
in the [config file](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/config.py) may need to be changed.

Then you need to run:

```
python setup_directory_structure.py
```

then you can either run individual scripts, or to run all three stages at
once ([binary_classifier.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/binary_classifier.py),
[classifier_consolidation.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/classifier_consolidation.py) or
[predict.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/predict.py)) you can run:

```
python main.py
```

**Outputs:**
The outputs from the code are in the [/data/output](https://github.com/crispinlogan/StarGalaxyQSOClassification/tree/master/data/output) directories. The specific directory
structure is created upon running [setup_directory_structure.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/setup_directory_structure.py), and the directory names are:
- RF - outputs fom RF gridsearch (best RF setups, and best metrics).
- hdbscan_gridsearch - outputs from hdbscan gridsearch (gridsearch labels data file, gridsearch performance summary datafile,
dendrograms, datapoints in PCA space)
- saved_models - saved models (scaler, PCA and hdbscan) trained on training data to later apply on new data.
- consolidation - outputs from consolidation (colour plots, metrics and summary of output labels, confusion matrices).
- prediction - outputs from prediction stage (colour plot, catalogue with predicted labels, summary of output labels,
datapoints in PCA space).

## Example / Test Run
We provide the [CPz.csv file](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/data/input/CPz.csv) (which is used in [LF2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract)) to
run the code on as an example. For the prediction stage, a random sample of
50,000 data points is selected from the KiDSVW catalogue (also presented in [LF2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract)),
and it is also provided [here](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/data/input/short_KiDSVW.csv).
This test can be run by setting `test_bool` in the [main.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/main.py) script to `True`.
For this test, the metrics obtained on the training catalogue (which will be found in
[/data/test_output](https://github.com/crispinlogan/StarGalaxyQSOClassification/tree/master/data/output) which is created upon running
`python setup_directory_structure.py`) should be as follows for the optimal consolidation method
(note the performance is sub-optimal, as the test runs a very quick gridsearch):

|             |   F1     | Accuracy  | Precision  |  Recall    | 
| :---        |  :----:  |   :----:  |    :----:  |    :----:  | 
| Star        | 0.9648   | 0.9892    | 0.9942     |  0.9371    | 
| Galaxy      | 0.98     | 0.9698    | 0.9828     |  0.9772    | 
| QSO         | 0.801    | 0.9702    | 0.9414     |  0.697     | 

However, we note that the training set provided, when run not in test mode (i.e.
when `test_bool` set to `False`), can achieve the following performance (again using the optimal consolidation method):

|             |   F1     | Accuracy  | Precision  |  Recall    | 
| :---        |  :----:  |   :----:  |    :----:  |    :----:  | 
| Star        | 0.9853   | 0.9954    | 0.9933     |  0.9775    | 
| Galaxy      | 0.9867   | 0.9799    | 0.9838     |  0.9896    | 
| QSO         | 0.9152   | 0.986     | 0.9568     |  0.8771    | 

The prediction stage is also run in the test setup on the [short_KiDSVW.csv](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/data/input/short_KiDSVW.csv) catalogue,
and there is a check in [main.py](https://github.com/crispinlogan/StarGalaxyQSOClassification/blob/master/hdbscan/pipeline/main.py) that the output from the prediction matches what is expected.
It is recommended to run this test to check that the scripts are running
as expected on your computer before running in the non-test setup, on your own data.

## Further Documentation
Further documentation can be found [here](https://crispinlogan.github.io/StarGalaxyQSOClassification)

## Further questions
If you have any questions about the code, please write to:
<span>crispin.logan@bristol.ac.uk</span> or 
<span>sotiria.fotopoulou@bristol.ac.uk</span>

## Citing
If you have used this codebase, please add a footnote with the link to this github repo (https://github.com/crispinlogan/StarGalaxyQSOClassification),
and cite the [LF2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A.154L/abstract) paper:

    @ARTICLE{2020A&A...633A.154L,
           author = {{Logan}, C.~H.~A. and {Fotopoulou}, S.},
            title = "{Unsupervised star, galaxy, QSO classification. Application of HDBSCAN}",
          journal = {\aap},
              year = 2020,
            month = jan,
           volume = {633},
            pages = {A154},
              doi = {10.1051/0004-6361/201936648}
    }
