class HelperFunctions():
    '''
    Holds helper functions.

    Args:
        ConfigObj (object): holds config variables

    Attributes:
        many! Aside from the Arg config variables object, all are methods/functions, and are detailed below.
    '''
    def __init__(self, ConfigObj):
        self.ConfigObj = ConfigObj


    def get_data(self, filename, filetype):
        '''
        Reads in catalogue data.

        Args:
            filename (str): name of catalgoue

            filetype (str): type of file (e.g. csv)

        Returns:
            incat (astropy table): table of catalogue data
        '''''
        from astropy.table import Table
        incat = Table.read(filename, format=filetype)
        return incat



    def get_all_features(self, indata, photo_band_list, combine_type, targetname = None, target = None):
        '''
        Creates attributes from catalogue data.

        Args:
            indata (astropy table): catalogue data - output from get_data function

            photo_band_list (list): list of photometric bands to use

            combine_type (str): subtract (for colours) or divide - how to make attributes
            from photometric bands

            targetname (str): column name of true labels

            target (str/int): corresponds to number of object type in true labels

        Returns:
            attribute_list (list): list of attribute values

            attribute_names (list): names of attributes

            attribute_target (array): binary labels for target type
        '''
        # verify photo_band_list exists in input file
        import sys
        import numpy as np
        for x in photo_band_list:
            if x not in indata.columns:
                print(indata.columns)
                sys.exit(x+' column not in file.')

        # construct attributes
        attribute_list = []
        attribute_names = []
        for i, attr1 in enumerate(photo_band_list):
            for j, attr2 in enumerate(photo_band_list):
                if j > i:

                    if combine_type == 'subtract':
                        attribute_list.append(
                            indata[attr1].data-indata[attr2].data)
                        attribute_names.append(
                            attr1+'-'+attr2)

                    elif combine_type == 'divide':
                        attribute_list.append(
                            indata[attr1].data/indata[attr2].data)
                        attribute_names.append(
                            attr1+'/'+attr2)

                    else:
                        sys.exit('Unknown '+combine_type+' attribute combination.')

        if targetname == None and target == None:
            return attribute_list, attribute_names

        else:
            attribute_target = np.where(indata[targetname].data == target, 1, 0)
            return attribute_list, attribute_names, attribute_target


    def do_scale(self, data, save = False):
        '''
        Scale attributes

        Args:
            data (array): input unscaled attributes

            save (bool): whether to save scaler or not

        Returns:
            scaled_data (array): scaled attributes
        '''
        from sklearn.preprocessing import StandardScaler
        import pickle
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        if save == True:
            model = scaler
            filename = self.ConfigObj.saved_scaler_model_file
            pickle.dump(model, open(filename, 'wb'))

        return scaled_data


    def do_random_forest(self, data, target, features, kind, gridsearch=True):
        '''
        Run random forest gridsearch to obtain best Random Forest hyperparameter
        setup (optional). Then runs best RF setup to give list of importances of each attribute.

        Args:
            data (array): input data of attributes and their values

            target (array): binary labels for target

            features (list): attribute names

            kind (str): object type star, galaxy, qso

            gridsearch (bool): whether to do gridsearch or not

        Returns:
            clf (object): RF trained classifier
        '''
        import json
        import numpy as np

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import f1_score, precision_score
        from sklearn.metrics import recall_score, accuracy_score
        from astropy.io import ascii

        if gridsearch == True:

            X_train, X_test, y_train, y_test = train_test_split(data, target, random_state = self.ConfigObj.random_state_val)

            param_grid = self.ConfigObj.RF_param_grid

            RFclassifier = RandomForestClassifier(max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  n_jobs=1,
                                                  random_state=self.ConfigObj.random_state_val,
                                                  verbose=0,
                                                  warm_start=False,
                                                  class_weight=None)

            clf = GridSearchCV(RFclassifier,
                               param_grid=param_grid,
                               cv=10,
                               n_jobs=self.ConfigObj.n_jobs)

            # fit the classifer with best parameters from GridSearch
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            f1 = f1_score(y_test, y_pred, average='binary')
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')

            params = {"random_state": self.ConfigObj.random_state_val,
                      "model_type": "RFclassifier",
                      "param_grid": str(clf.best_params_),
                      "stratify": False
                      }
            metrics = {"f1": f1,
                       "accuracy": accuracy,
                       "recall": recall,
                       "precision": precision
                       }
            print(params)
            print()
            print(metrics)

            with open(self.ConfigObj.RF_best_params_file.replace('CLASS', kind), 'w') as f1:
                json.dump(clf.best_params_, f1)

            with open(self.ConfigObj.RF_best_metrics_file.replace('CLASS', kind), 'w') as f1:
                json.dump(metrics, f1)

            importances = clf.best_estimator_.feature_importances_


        else:
            with open(self.ConfigObj.RF_best_params_file.replace('CLASS', kind)) as f2:
                params = json.load(f2)
                print(params)
                #best_params = params['param_grid']
            #print(best_params)

            clf = RandomForestClassifier(**params)

            clf.fit(data, target)

            importances = clf.feature_importances_

        idx = importances.argsort()[::-1]

        ascii.write([np.array(features)[idx], np.array(importances)[idx]],
                    self.ConfigObj.RF_importances.replace('CLASS', kind),
                    names=['#feature', 'importance'],
                    overwrite=True)
        return clf


    def select_important_attributes(self, attribute_names, RF_importances_file, top=0):
        '''
        Select attributes according to their importance (from Random Forest output).

        Args:
            attribute_names (list): attribute names

            RF_importances_file (str): filepath for RF imprtances file (output from do_random_forest function)

            top (int): number of top attributes to select

        Returns:
            index (list): indices of selected attributes
        '''
        # top=-1 keeps non-zero importances
        # top=N keeps N top rank importances
        import numpy as np

        ## read RF_importances
        feature = []
        importan = []

        with open(RF_importances_file, 'r') as f:
            RF = f.read().splitlines()

        for line in RF:
            if line[0]=='#':
                continue
            feature.append(line.split()[0])
            importan.append(float(line.split()[1]))

        feature_name = np.array(feature)
        importances = np.array(importan)

        idx = importances.argsort()[::-1]

        if top == 0:
            important_idx = np.nonzero(importances[idx])[0]
        else:
            important_idx = idx[:top]

        # find location of important attributes in 'attribute_names'
        names = feature_name[important_idx]

        index = [ list(attribute_names).index(n) for n in names]
        return index


    def do_pca(self, data, ncomp=3, kind = None, save = False):
        '''
        Run PCA decomposition on data.

        Args:
            data (array): data on which PCA is to be performed

            ncomp (int): PCA reduction dimension

            kind (str): object type - star, galaxy, qso (only set if save set to True)

            save (bool): whether to save PCA reducer or not

        Returns:
            fitted_pca (object): fitted PCA reducer model
        '''
        from sklearn.decomposition import PCA
        import pickle
        pca = PCA(n_components=ncomp, random_state = self.ConfigObj.random_state_val)
        fitted_pca = pca.fit(data)

        if save == True:
            model = fitted_pca
            filename = self.ConfigObj.saved_pca_model_file.replace('CLASS', kind)
            pickle.dump(model, open(filename, 'wb'))

        return fitted_pca


    def do_hdbscan_gridsearch(self, attribute_names, scaled_data, kind):
        '''
        Run hdbscan hyperparameter gridsearch to get best hyperparameter setup
        to be used later in final model.

        Args:
            attribute_names (list): list of attribute names

            scaled_data (array): scaled attributes

            kind (str): object type - star, galaxy, qso

        Returns:
            cluster_labels (array): output labels according to clustering result

            clusterer (object): trained hdbscan clusterer
        '''
        import hdbscan
        import numpy as np

        ## RF_top, PCA, min_cluster_size gridsearch
        c = []
        column_names = []
        for RF_top in self.ConfigObj.RF_top_gridsearch_vals:
            for ncomp in self.ConfigObj.ncomp_gridsearch_vals:
                for min_cluster_size in self.ConfigObj.min_cluster_size_gridsearch_vals:
                    ## PCA
                    important_idx = self.select_important_attributes(attribute_names, self.ConfigObj.RF_importances.replace('CLASS',kind), top=RF_top)

                    important_data = scaled_data[:, important_idx]

                    pca = self.do_pca(important_data, ncomp=ncomp, kind=kind, save = False)
                    pca_components = pca.transform(important_data)

                    ##################################################
                    ## HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            core_dist_n_jobs=self.ConfigObj.n_jobs)

                    cluster_labels = clusterer.fit_predict(pca_components)
                    c.append(list(cluster_labels))
                    column_names.append('{}_{}_{}'.format(RF_top, ncomp, min_cluster_size))

        np.savetxt(self.ConfigObj.HDBSCAN_gridsearch.replace('CLASS', kind), np.array(c).T, header=' '.join(column_names))

        return cluster_labels, clusterer





    def train_and_save_hdbscan(self, attribute_names, scaled_data, hdbscan_setup, kind):
        '''
        Trains and saves an hdbscan clusterer object for use later in predict stage.
        Also saves the position in PCA space for each datapoint to a text file.
        Also saves a dendrogram plot from the trained hdbscan clusterer.

        Args:
            attribute_names (list): Names of all possible attributes

            scaled_data (array): Scaled colour data

            hdbscan_setup (str): Contains setup for hdbscan training in form of
            '{}_{}_{}'.format(RF_top, ncomp, min_cluster_size)

            kind (str): object type (star, galaxy, or qso)

        Returns:

        '''
        import hdbscan
        import pickle
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        ## Read best RF_top, PCA, min_clustersize from input string
        RF_top = int(hdbscan_setup.split('_')[0])
        ncomp = int(hdbscan_setup.split('_')[1])
        min_cluster_size = int(hdbscan_setup.split('_')[2])

        important_idx = self.select_important_attributes(attribute_names, self.ConfigObj.RF_importances.replace('CLASS',kind), top=RF_top)

        important_data = scaled_data[:, important_idx]

        pca = self.do_pca(important_data, ncomp=ncomp, kind=kind, save = True)
        pca_components = pca.transform(important_data)
        #save pca components to file
        np.savetxt(self.ConfigObj.PCA_dimensions_file.replace('CLASS',kind), pca_components)

        ##################################################
        ## HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    core_dist_n_jobs=self.ConfigObj.n_jobs,prediction_data = True)

        model = clusterer.fit(pca_components)

        filename = self.ConfigObj.saved_hdbscan_model_file.replace('CLASS',kind)
        pickle.dump(model, open(filename, 'wb'))

        # save dendrogram
        clusterer.condensed_tree_.plot(select_clusters=True,selection_palette=sns.color_palette('deep', 8))
        plt.tight_layout()
        plt.savefig(self.ConfigObj.dendrogram_file.replace('CLASS',kind))
        plt.clf()




    def compute_performances(self, indata, kind):
        '''
        Computes classification metrics from all labels from hdbscan gridsearch,
        and produces a file with the metrics for each setup.

        Args:
            indata (astropy table): catalogue data - output from get_data function

            kind (str): object type (star, galaxy, or qso)

        Returns:

        '''
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        from astropy.io import ascii
        # labels from input data
        # target label for binary classifier, set in config.py
        target_class = self.ConfigObj.hclass_dict[kind]
        target_mask = indata[self.ConfigObj.targetname]==target_class
        # transform data class into binary labels
        y_true = target_mask.astype(int)

        with open(self.ConfigObj.HDBSCAN_gridsearch.replace('CLASS', kind),'r') as f:
            hdb_runs = np.genfromtxt(f, names=True)

        # column names correspond to the HDBSCAN setup RFtop_Npca_MinClusterSize
        hdb_names = hdb_runs.dtype.names

        # HDBSCAN splits the sample in a number of classes
        # For each RFtop_Npca_MinClusterSize, loop through all clusters as if
        # they were binary classifiers and evaluate performance
        # record in file the best cluster number and its performance
        out_name = []
        out_Nclusters = []
        out_best_cluster = []
        out_f1 = []
        out_acc = []
        out_prec = []
        out_rec = []
        for name in hdb_names:
            # unique clusters in setup
            Nclusters = np.unique(hdb_runs[name])
            out_name.append(name)
            out_Nclusters.append(len(Nclusters))

            # calculate f1 score for each of the clusters
            group_f1 = []
            for cluster in Nclusters:
                cluster_mask = hdb_runs[name] == cluster
                y_pred = cluster_mask.astype(int)
                group_f1.append( f1_score(y_true, y_pred) )

            # best f1 score cluster is saved
            max_loc = np.argmax(np.array(group_f1))

            max_f1 = group_f1[max_loc]
            max_cl = Nclusters[max_loc]

            out_best_cluster.append(max_cl)
            out_f1.append(max_f1)

            # compute performance metrics for best cluster
            out_cluster_mask = hdb_runs[name] == max_cl
            y_pred = out_cluster_mask.astype(int)

            out_acc.append(accuracy_score(y_true,  y_pred))
            out_prec.append(precision_score(y_true,  y_pred))
            out_rec.append(recall_score(y_true,  y_pred))

        # return the labels and HDBSCAN setup for absolute max F1 score in table
        f1_loc = np.argmax(np.array(out_f1))

        # save best cluster metrics per run
        performance = [out_name, out_Nclusters, out_best_cluster, out_f1, out_acc, out_prec, out_rec]

        column_names = ['#HDBSCAN_setup', 'Nclusters', 'best_cluster', 'F1', 'accuracy', 'precision', 'recall']
        ascii.write(performance, self.ConfigObj.HDBSCAN_gridsearch_performance.replace('CLASS',kind), names=column_names, overwrite=True)




    def find_best_hdbscan_setup(self, performance_dat_file):
        '''
        Finds best hdbscan setup for given object

        Args:
            performance_dat_file (string): path and name of performance dat file

        Returns:
            best_name (string): column name/setup for best hdbscan

            best_cluster (int/string): hdbscan cluster value
        '''
        import pandas as pd
        import numpy as np
        perf_df = pd.read_csv(performance_dat_file, delim_whitespace=True)
        best_name = np.array(perf_df.sort_values(by=['F1'],ascending=False)['#HDBSCAN_setup'])[0]
        best_cluster = np.array(perf_df.sort_values(by=['F1'],ascending=False)['best_cluster'])[0]
        return best_name, int(best_cluster)




    def write_best_labels_binary(self, best_name,best_cluster,kind):
        '''
        Writes best labels to file in a binary format

        Args:
            best_name (string): column name/setup for best hdbscan

            best_cluster (int/string): hdbscan cluster value

            kind (str): object type (star, galaxy, or qso)

        Returns:
        '''
        #select col with predicted labels from gridsearch with optimal setup
        import pandas as pd
        with open(self.ConfigObj.HDBSCAN_gridsearch.replace('CLASS', kind), "r") as file:
            first_line = file.readline()
            col_names = first_line.rstrip().split(' ')[1:]
        gridsearch_output = pd.read_csv(self.ConfigObj.HDBSCAN_gridsearch.replace('CLASS', kind),
                            delim_whitespace=True,names=col_names,comment='#')
        predicted_labels = gridsearch_output[best_name].copy()
        del gridsearch_output #to free up the memory as file not needed again
        #turn predicted labels into 1 for obj and 0 for not obj
        predicted_labels[predicted_labels != best_cluster] = -99
        predicted_labels[predicted_labels == best_cluster] = 1
        predicted_labels[predicted_labels == -99] = 0
        #
        predicted_labels.to_csv(self.ConfigObj.hdbscan_best_labels.replace('CLASS',kind), index=False, header=False)




    def plot_classification(self, indata, labels1, labels2, labels1_name, labels2_name, dict_labels1, dict_labels2):
        '''
        Creates two side-by-side colour plots for two sets of labels for the same catalogue.

        Args:
            indata (astropy table): catalogue data - output from get_data function

            labels1 (array): First set of labels

            labels2 (array): Second set of labels

            labels1_name (str): Name of first set of labels to appear in plot

            labels2_name (str): Name of second set of labels to appear in plot

            dict_labels1 (dict): Dict with keys as object name, value as numeric label value
            for first set of labels

            dict_labels2 (dict): Dict with keys as object name, value as numeric label value
            for second set of labels

        Returns:
            fig, ax (objects): fig and ax objects of plot as from matplotlib
        '''
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2)
        gs.update(left=0.08, right=0.95, wspace=0.200,
                  bottom=0.1, top=0.9, hspace=0.20)

        ax = plt.subplot(gs[0, 0])

        for obj in ['galaxy','star','qso','outlier']:
            if not (obj in dict_labels1.keys()):
                continue
            mask = [labels1 == dict_labels1[obj]]
            plt.scatter(indata['g'][tuple(mask)]-indata['j'][tuple(mask)],
                           indata['y'][tuple(mask)]-indata['w1'][tuple(mask)],
                           c=self.ConfigObj.plot_colours_dict[obj], s=5)
        plt.xlim([-4, 8])
        plt.xlabel('g-J')
        plt.ylabel('y-W1')
        plt.title(labels1_name)

        ax = plt.subplot(gs[0, 1])
        for obj in ['galaxy','star','qso','outlier']:
            if not (obj in dict_labels2.keys()):
                continue
            mask = [labels2 == dict_labels2[obj]]
            plt.scatter(indata['g'][tuple(mask)]-indata['j'][tuple(mask)],
                          indata['y'][tuple(mask)]-indata['w1'][tuple(mask)],
                          c=self.ConfigObj.plot_colours_dict[obj], s=5, label = obj)
        plt.title(labels2_name)
        plt.xlabel('g-J')
        plt.ylabel('y-W1')
        plt.xlim([-4, 8])
        plt.legend()

        return fig, ax




    def find_object_indices(self, predicted_labels_dict):
        '''
        Finds indices in arrays for objects and combinations / duplicates of
    positive classifications from the binary classifiers.
        Required for do_consolidation (optimal or alternative) functions.

        Args:
            predicted_labels_dict (dict): dict of arrays with each binary classifier's
            predicted labels (with 1 for positive classification, 0 otherwise)

        Returns:
            list_of_indices (list of arrays): various arrays with different indices

            before_consolidation_str (string): number of objs pre-consolidation
        '''
        import numpy as np

        #find indices for each binary classifier positive identification
        star_indices = np.where(predicted_labels_dict['star']==1)[0]
        gal_indices =  np.where(predicted_labels_dict['galaxy']==1)[0]
        qso_indices =  np.where(predicted_labels_dict['qso']==1)[0]

        before_consolidation_str = f"Pre-classification, positively classified points \
    for binary classifiers is: star: {len(star_indices)}, gal: {len(gal_indices)} \
    qso: {len(qso_indices)}.\n"

        #find common indices - i.e. indices where an object is classified as e.g. both star and gal...
        star_gal_indices = np.intersect1d(star_indices,gal_indices)
        star_qso_indices = np.intersect1d(star_indices,qso_indices)
        qso_gal_indices =  np.intersect1d(qso_indices,gal_indices)

        #find unique combination of the above 3 arrays (i.e. no duplicates)
        all_indices = np.unique(np.append(star_gal_indices,
                                          np.append(star_qso_indices,qso_gal_indices)))

        list_of_indices = list([star_indices, gal_indices, qso_indices, star_gal_indices,
                                star_qso_indices, qso_gal_indices, all_indices])
        return list_of_indices, before_consolidation_str



    def do_consolidation(self, predicted_labels_dict,list_of_indices, consolidation_type):
        '''
        Use optimal consolidation method to consolidate binary classifiers labels.
        Also writes some results to a file.

        Args:
            predicted_labels_dict (dict): dict of arrays with each binary classifier's
            predicted labels (with 1 for positive classification, 0 otherwise)

            list_of_indices (list of arrays): various arrays with different indices -
            see find_object_indices()

            consolidation_type (string): 'optimal' or 'alternative' depending on which
            consolidation method to use

        Returns:
            cluster_labels (array): consolidated labels

            after_cons_str (string): number of objs post-consolidation
        '''
        import copy
        import numpy as np

        star_indices, gal_indices, qso_indices, star_gal_indices,star_qso_indices, \
            qso_gal_indices, all_indices = list_of_indices

        predicted_labels_dict_copy = copy.deepcopy(predicted_labels_dict)

        cluster_labels = np.full(len(predicted_labels_dict_copy['star']),self.ConfigObj.hdbscanclass_dict['outlier'])
        cluster_labels[gal_indices] = self.ConfigObj.hdbscanclass_dict['galaxy']
        cluster_labels[star_indices] = self.ConfigObj.hdbscanclass_dict['star']
        cluster_labels[qso_indices] = self.ConfigObj.hdbscanclass_dict['qso']
        if consolidation_type == 'alternative':
            cluster_labels[all_indices] = self.ConfigObj.hdbscanclass_dict['outlier']

        after_cons_str = f"{consolidation_type} consolidation method: star: \
    {sum(cluster_labels == self.ConfigObj.hdbscanclass_dict['star'])}, \
    gal: {sum(cluster_labels == self.ConfigObj.hdbscanclass_dict['galaxy'])}, \
    qso: {sum(cluster_labels == self.ConfigObj.hdbscanclass_dict['qso'])}, \
    outlier: {sum(cluster_labels == self.ConfigObj.hdbscanclass_dict['outlier'])}.\n"

        return cluster_labels, after_cons_str





    def compute_metric_scores(self, true_labels, pred_labels):
        '''
        Computes F1, Acc., Prec, Rec. for each of the object classes (star, galaxy,
    qso), from consolidated labels. N.B. the metric scores are for each of the
    classes one at a time.

        Args:
            true_labels (array): true labels

            pred_labels (array): predicted labels

        Returns:
            results_str (string): string of metric scores for each object

            metrics_list (list): summarizes/packages up metric scores
        '''
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        f1_vals = []
        acc_vals = []
        prec_vals = []
        rec_vals = []
        no_objs_pred = []
        results_str = ''
        for obj in self.ConfigObj.classes:
            true = np.zeros(len(true_labels))
            pred = np.zeros(len(pred_labels))
            true[true_labels==self.ConfigObj.hdbscanclass_dict[obj]] = 1
            pred[pred_labels==self.ConfigObj.hdbscanclass_dict[obj]] = 1
            f1        = round( f1_score(true,pred,average='binary') , 4 )
            accuracy  = round( accuracy_score(true,pred) , 4)
            precision = round( precision_score(true,pred,average='binary') , 4 )
            recall =    round( recall_score(true,pred,average='binary') , 4 )
            results_str+=f"Results for {obj}: \tF1:{f1}, \tAcc:{accuracy}, \tPrec:{precision}, \tRec:{recall}\n"
            f1_vals.append(f1)
            acc_vals.append(accuracy)
            prec_vals.append(precision)
            rec_vals.append(recall)
            no_objs_pred.append(pred.sum())
        metrics_list = [f1_vals, acc_vals, prec_vals, rec_vals, no_objs_pred]
        return results_str, metrics_list



    def plot_confusion_matrix(self, y_true, y_pred, identifiers, normalize = True):
        '''
        Plot confusion matrix

        Args:
            y_true (array): True labels

            y_pred (array): Predicted labels

            f (list): List of names of objects that the values in the labels arrays
            refer to in order of size of value in label arrays. If length of list
            is 1, then will convert input labels to binary form and plot a binary
            confusion matrix

            normalize (bool): Sets whether we normalize confusion matrix or not

        Returns:
            fig (object): Plot object

            ax (str): Plot object
        '''
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import numpy as np
        pred_labels = y_pred.copy()
        true_labels = y_true.copy()
        #loop effectively turns multi-class labels into binary labels
        #(could do as separate function?)
        if len(identifiers) == 1:
            obj = identifiers[0]
            #make binary
            mask_pred = (pred_labels==self.ConfigObj.hdbscanclass_dict[f'{obj}'])
            pred_labels[pred_labels==pred_labels] = -99
            pred_labels[mask_pred] = 0
            pred_labels[np.invert(mask_pred)] = 1
            #
            mask_true = (true_labels==self.ConfigObj.hclass_dict[f'{obj}'])
            true_labels[true_labels==true_labels] = -99
            true_labels[mask_true] = 0
            true_labels[np.invert(mask_true)] = 1
            #
            identifiers = [f'{obj}',f'not {obj}']
        #
        if normalize == True:
            A = np.round( confusion_matrix(true_labels, pred_labels, normalize='true'), 2 )
        else:
            A = confusion_matrix(true_labels, pred_labels)
        #
        fig, ax = plt.subplots()
        im = ax.imshow(A,cmap=plt.cm.Blues)
        #
        ax.set_xticks(np.arange(len(identifiers)))
        ax.set_yticks(np.arange(len(identifiers)))
        ax.set_xticklabels(identifiers)
        ax.set_yticklabels(identifiers)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        #
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = "anchor")
        #
        #loop over data and create annotations
        for i in range(len(identifiers)):
            for j in range(len(identifiers)):
                #text = ax.text(j,i,A[i,j],
                #               ha='center',va='center',weight='bold', size = 12)#,color='w')
                ax.text(j,i,A[i,j],
                               ha='center',va='center',weight='bold', size = 12)#,color='w')
        #
        fig.colorbar(im)
        fig.tight_layout()
        #
        return fig, ax
