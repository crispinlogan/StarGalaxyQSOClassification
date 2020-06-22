from config import ConfigVars
from helper_functions import HelperFunctions
import hdbscan
import numpy as np
import pickle
import astropy.io.ascii


def run_predict(conf, lib):
    '''
    Wrapper function for predict code.

    Args:
        conf (object): contains config variables

        lib (object): contains helper functions

    Returns:
        after_opt_str (str): info on optimal consolidation output (used to check
        test run has run successfully)

    '''''
    indata = lib.get_data(conf.data_file_predicted, 'csv')## construct features, subtract (mag) or divide (fluxes)

    ## construct features, subtract (mag) or divide (fluxes)
    attribute_list, attribute_names = lib.get_all_features(indata, conf.photo_band_list, conf.combine_type)

    ## whitening & normalization of data - i.e. scaling here (note should go in loop below)
    loaded_scaler_model =  pickle.load(open(conf.saved_scaler_model_file, 'rb'))
    new_data_scaled = loaded_scaler_model.transform(np.transpose(np.array(attribute_list)))

    dict_predicted_labels = {}
    for obj in conf.classes:
        #load pre-trained models
        pca_filename = conf.saved_pca_model_file.replace('CLASS',obj)
        hdbscan_filename = conf.saved_hdbscan_model_file.replace('CLASS',obj)
        loaded_pca_model =  pickle.load(open(pca_filename, 'rb'))
        loaded_hdbscan_model = pickle.load(open(hdbscan_filename, 'rb'))
        #
        best_name, best_cluster = lib.find_best_hdbscan_setup(conf.HDBSCAN_gridsearch_performance.replace('CLASS',obj))
        RF_top = int(best_name.split('_')[0])
        #
        important_idx = lib.select_important_attributes(attribute_names, conf.RF_importances.replace('CLASS',obj), top=RF_top)
        important_data = new_data_scaled[:, important_idx].copy()
        #
        #pca
        new_data_scaled_and_pca = loaded_pca_model.transform(important_data)
        #save pca components to file
        np.savetxt(conf.PCA_dimensions_file_prediction_data.replace('CLASS', obj), new_data_scaled_and_pca)
        #
        #predict labels of new catalogue
        predicted_labels = hdbscan.approximate_predict(loaded_hdbscan_model, new_data_scaled_and_pca)[0]
        #
        #turn predicted labels into 1 for obj and 0 for not obj - i.e. binary-ify labels!
        predicted_labels[predicted_labels != best_cluster] = -99
        predicted_labels[predicted_labels == best_cluster] = 1
        predicted_labels[predicted_labels == -99] = 0
        #
        dict_predicted_labels[obj] = predicted_labels.copy()
        #
        #append new binary labels to input new catalogue
        indata[f'{obj}_binary_labels_predicted'] = predicted_labels.copy()


    #do consolidation - very similar to clasifier_consolidation...
    fh_op = open(conf.HDBSCAN_consolidation_summary_predicted, 'w+')

    list_of_indices, before_consolidation_str = lib.find_object_indices(dict_predicted_labels)
    print(before_consolidation_str)

    optimal_labels, after_opt_str = lib.do_consolidation(dict_predicted_labels,list_of_indices,'optimal')
    alternative_labels, after_alt_str = lib.do_consolidation(dict_predicted_labels,list_of_indices,'alternative')

    print(after_opt_str)
    fh_op.write(after_opt_str)
    print(after_alt_str)
    fh_op.write(after_alt_str)

    fh_op.close()

    #append consolidated labels to new catalogue and save new catalogue with
    #binary and consolidated labels
    indata['optimal_labels_predicted'] = optimal_labels
    indata['alternative_labels_predicted'] = alternative_labels

    ## save table with labels to file
    astropy.io.ascii.write(indata, output = conf.Catalogue_with_hdbscan_labels_predicted, format = 'csv', overwrite=True)

    #plot results - plot subset if huge dataset - currently 50k maximum
    fig, ax = lib.plot_classification(indata[:50000], optimal_labels[:50000], alternative_labels[:50000], 'optimal labels', ' alternative labels', conf.hdbscanclass_dict, conf.hdbscanclass_dict)
    fig.savefig(conf.pred_colour_plot)

    return after_opt_str


if __name__ == '__main__':
    conf = ConfigVars(test = False)
    lib = HelperFunctions(conf)
    run_predict(conf, lib)
