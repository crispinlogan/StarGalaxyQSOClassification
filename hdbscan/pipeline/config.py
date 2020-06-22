import os

class ConfigVars():
    '''
    Holds config variables. test (see Args) has to be passed on instantiation.
    The other three args are optional. A lot of other attributes are automatically set
    on instantiation (see Attributes). Note that some of the attributes are 'generic',
    and the CLASS in the string may be replaced in other scripts with e.g. 'star' or 'galaxy'
    etc.

    Args:
        test (bool): whether test run or not

        kind (str): set to object type (star, galaxy, or qso)

        gridsearch (bool): whether to run Random Forest (RF) gridsearch or not

        n_jobs (int): no of cores to run on. Automatically set to half of available cores

    Attributes:
        test (bool): whether test run or not

        kind (str): set to object type (star, galaxy, or qso)

        gridsearch (bool): whether to run Random Forest gridsearch or not

        n_jobs (int): no of cores to run on. Automatically set to half of available cores


        base_path (str): base path reference for other paths

        random_state_val (int): used for non-deterministic steps (PCA, RF)

        catname (str): name of input catalogue (with labels)

        catname_predicted (str): name of unsen catalogue for prediction

        targetname (str): column name of catname that contains the true labels

        classes (list): object types for classification

        photo_band_list (list): list of photometric bands used to create the
        attributes (see combine_type)

        combine_type (str): 'subtract' or 'divide' - sets how to combine the
        photometric bands (i.e. 'subtract' creates colours)

        hclass_dict (dict): mapping of numeric values in targetname column to
        object type

        hdbscanclass_dict (dict): numeric labels to be used for HDBSCAN
        predicted object types

        plot_colours_dict (dict): object to colour dict used for plotting

        RF_param_grid (list of dict): RF parameter grid to be used for RF gridsearch

        RF_top_gridsearch_vals (list): top features from ranked RF importances to
        be used to select features to hdbscan gridsearch

        ncomp_gridsearch_vals (list): number of dimensions to which to reduce
        using PCA for input to hdbscan gridsearch

        min_cluster_size_gridsearch_vals (range / list): min_cluster_size values
        to try in hdbscan gridsearch

        data_output (str): path name for data output

        data_file_predicted (str): path to catalogue for prediction

        RF_dir (str): path to RF output

        hdb_dir (str): path to hdbscan output

        sav_mod_dir (str): path to saved models

        cons_dir (str): path to consolidation step output

        pred_dir (str): path to prediction step output

        data_file (str): path to catalogue for training data with labels

        RF_best_params_file (str): path to best hyperparameters for RF from RF gridsearch

        RF_best_metrics_file (str): path to best metrics for best RF from RF gridsearch

        RF_importances (str): path to list of feature importances from RF

        HDBSCAN_gridsearch (str): path to output file from hdbscan gridsearch (labels for each stup)

        HDBSCAN_gridsearch_performance (str): path to summary of metrics from
        each setup from hdbscan gridsearch

        dendrogram_file (str): path to dendrogram plot from best hdbscan setup

        hdbscan_best_setups_file (str): path to summary file of best hdbscan classifier setup

        PCA_dimensions_file (str): path to positions in PCA space for each datapoint in training dataset

        saved_scaler_model_file (str): path to saved scaler model

        saved_pca_model_file (str): path to saved PCA model

        saved_hdbscan_model_file (str): path to saved hdbscan model

        HDBSCAN_consolidation_summary (str): path to file with summary of
        classification performance

        Catalogue_with_hdbscan_labels (str): path to catalogue with new hdbscan labels appended

        hdbscan_best_labels (str): path to best labels for each hdbscan binary classifier

        confusion_plots (str): path to confusion plots for consolidated labels

        opt_colour_plot (str): path to colour plot of consolidated lables (using optimal consolidation)

        alt_colour_plot (str): path to colour plot of consolidated lables (using alternative consolidation)

        HDBSCAN_consolidation_summary_predicted (str): path to summary of labels for label prediction on unseen catalogue

        Catalogue_with_hdbscan_labels_predicted (str): path to unseen catalogue with predicted labels

        pred_colour_plot (str): path to colour plot for predicted labels on unseen catalogue

        PCA_dimensions_file_prediction_data (str): path to positions in PCA space for each datapoint in prediction dataset

    '''
    def __init__(self, test, kind = None, gridsearch=False, n_jobs = max(int(os.cpu_count()/2),1)):
        self.test = test # True, False
        self.kind = kind # 'star','galaxy','qso'
        self.gridsearch = gridsearch
        self.n_jobs = n_jobs

        self.base_path = '' # your path goes here
        assert self.base_path , 'base_path is not set: Set base_path in config file'

        self.random_state_val = 1

        self.catname = 'CPz'
        self.catname_predicted = 'short_KiDSVW'

        self.targetname = 'hclass'

        self.classes = ['star','galaxy','qso']

        #suffix *3 means 3'' aperture, otherwise it's total magnitude
        self.photo_band_list = ['u', 'g', 'r', 'i', 'z', 'y', 'j', 'h', 'k',
                        'u3', 'g3', 'r3', 'i3', 'z3', 'y3', 'j3', 'h3', 'k3',
                        'w1', 'w2']

        # subtract or divide to get features
        self.combine_type = 'subtract'

        self.hclass_dict = {'unknown': -1,
                       'star': 0,
                       'galaxy': 1,
                       'agn': 2,
                       'qso': 3}

        #vals most similar to Fotopoulou18
        self.hdbscanclass_dict = {'outlier': -1,
                       'star': 0,
                       'galaxy': 1,
                       'qso': 3}

        #colours to be used in plots
        self.plot_colours_dict = {'outlier':'b',
                                  'star':'#000000',
                                  'galaxy':'#E69F00',
                                  'qso':'#56B4E9'}

        if self.test == False:
            #hyperparams for RF gridsearch
            self.RF_param_grid = [{'n_estimators': [100, 150, 200],
                              'criterion':['entropy'],
                              'max_depth': [30],
                              'min_samples_split': [3, 4, 5],
                              'min_samples_leaf': [1, 2],
                              'max_features':['log2', 15, 30],
                              'bootstrap': [True],
                              'oob_score': [True]}]
            #hyperparams for hdbscan gridsearch
            self.RF_top_gridsearch_vals = [0, 50, 30, 20, 10, 5]
            self.ncomp_gridsearch_vals = [2, 3, 4, 5]
            self.min_cluster_size_gridsearch_vals = range(3, 100)
            #where to save output data
            self.data_output = 'data/output'
            #new catalogue to predict on
            self.data_file_predicted = os.path.join(self.base_path,f'data/input/{self.catname_predicted}.csv')

        elif self.test == True:
            #hyperparams for RF gridsearch
            self.RF_param_grid = [{'n_estimators': [50],
                              'criterion':['entropy', 'gini'],
                              'max_depth': [20],
                              'min_samples_split': [3],
                              'min_samples_leaf': [1],
                              'max_features':[20],
                              'bootstrap': [True],
                              'oob_score': [True]}]
            #hyperparams for hdbscan gridsearch
            self.RF_top_gridsearch_vals = [10]
            self.ncomp_gridsearch_vals = [3]
            self.min_cluster_size_gridsearch_vals = range(45, 50)
            #where to save output data
            self.data_output = 'data/test_output'
            #new catalogue to predict on
            self.data_file_predicted = os.path.join(self.base_path,'data/input/short_KiDSVW.csv')


        #Output directories
        self.RF_dir = f'{self.base_path}/{self.data_output}/RF'
        self.hdb_dir = f'{self.base_path}/{self.data_output}/hdbscan_gridsearch'
        self.sav_mod_dir = f'{self.base_path}/{self.data_output}/saved_models'
        self.cons_dir = f'{self.base_path}/{self.data_output}/consolidation'
        self.pred_dir = f'{self.base_path}/{self.data_output}/prediction'

        #Input data file (note the data_file_predicted is set above)
        self.data_file = os.path.join(self.base_path,f'data/input/{self.catname}.csv')

        #RF gridsearch output
        self.RF_best_params_file = f'{self.RF_dir}/{self.catname}_RF_best_params_CLASS.json'
        self.RF_best_metrics_file = f'{self.RF_dir}/{self.catname}_RF_best_metrics_CLASS.json'
        self.RF_importances = f'{self.RF_dir}/{self.catname}_RF_importances_CLASS.dat'

        #HDBSCAN (gridsearch) output
        self.HDBSCAN_gridsearch = f'{self.hdb_dir}/{self.catname}_HDBSCAN_gridsearch_CLASS.dat'
        self.HDBSCAN_gridsearch_performance = f'{self.hdb_dir}/{self.catname}_HDBSCAN_gridsearch_performance_CLASS.dat'
        self.dendrogram_file = f'{self.hdb_dir}/{self.catname}_HDBSCAN_dendrogram_CLASS.png'
        self.hdbscan_best_setups_file = f'{self.hdb_dir}/{self.catname}_HDBSCAN_best_setup_CLASS.txt'
        self.PCA_dimensions_file = f'{self.hdb_dir}/{self.catname}_PCA_dimensions_CLASS.csv'

        #Saved models
        self.saved_scaler_model_file = f'{self.sav_mod_dir}/{self.catname}_saved_scaler_model.sav'
        self.saved_pca_model_file = f'{self.sav_mod_dir}/{self.catname}_saved_pca_model_CLASS.sav'
        self.saved_hdbscan_model_file = f'{self.sav_mod_dir}/{self.catname}_saved_hdbscan_model_CLASS.sav'

        #Consolidation output
        self.HDBSCAN_consolidation_summary = f'{self.cons_dir}/{self.catname}_consolidation_summary.txt'
        self.Catalogue_with_hdbscan_labels = f'{self.cons_dir}/{self.catname}_with_hdbscan_labels.csv'
        self.hdbscan_best_labels = f'{self.cons_dir}/{self.catname}_hdbscan_best_labels_CLASS.csv'
        self.confusion_plots = f'{self.cons_dir}/{self.catname}_confusion_plot_CLASS.png'
        self.opt_colour_plot = f'{self.cons_dir}/{self.catname}_optimal_colour_plot.png'
        self.alt_colour_plot = f'{self.cons_dir}/{self.catname}_alternative_colour_plot.png'

        #Prediction output
        self.HDBSCAN_consolidation_summary_predicted = f'{self.pred_dir}/{self.catname_predicted}_consolidation_summary.txt'
        self.Catalogue_with_hdbscan_labels_predicted = f'{self.pred_dir}/{self.catname_predicted}_with_hdbscan_labels.csv'
        self.pred_colour_plot = f'{self.pred_dir}/{self.catname_predicted}_predicted_colour_plot.png'
        self.PCA_dimensions_file_prediction_data = f'{self.pred_dir}/{self.catname_predicted}_PCA_dimensions_prediction_data_CLASS.csv'
