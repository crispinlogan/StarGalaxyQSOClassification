from config import ConfigVars
from helper_functions import HelperFunctions
import numpy as np
import pandas as pd
import astropy.io.ascii


def run_consolidation(conf, lib):
    '''
    Wrapper function for classifier consolidation code.

    Args:
        conf (object): contains config variables

        lib (object): contains helper functions

    Returns:

    '''''
    ## read in data
    indata = lib.get_data(conf.data_file, 'csv')

    dict_predicted_labels = {}

    for obj in conf.classes:
        #get best labels
        predicted_labels = pd.read_csv(conf.hdbscan_best_labels.replace('CLASS',f'{obj}'), header=None)
        predicted_labels = np.array(predicted_labels).reshape(len(predicted_labels),)    #
        dict_predicted_labels[obj] = predicted_labels
        # append column to input file with each classification result
        indata[f'{obj}_binary_labels'] = predicted_labels

    ## do consolidation
    fh_op = open(conf.HDBSCAN_consolidation_summary, 'w+')

    list_of_indices, before_consolidation_str = lib.find_object_indices(dict_predicted_labels)
    print(before_consolidation_str)

    optimal_labels, after_opt_str = lib.do_consolidation(dict_predicted_labels,list_of_indices,'optimal')
    alternative_labels, after_alt_str = lib.do_consolidation(dict_predicted_labels,list_of_indices,'alternative')

    print(after_opt_str)
    fh_op.write(after_opt_str)
    print(after_alt_str)
    fh_op.write(after_alt_str)

    #plot consolidated labels
    fig, ax = lib.plot_classification(indata, indata[f'{conf.targetname}'],optimal_labels,'spec labels','optimal labels', conf.hclass_dict, conf.hdbscanclass_dict)
    fig.savefig(conf.opt_colour_plot)
    fig, ax = lib.plot_classification(indata, indata[f'{conf.targetname}'],alternative_labels,'spec labels','alternative labels', conf.hclass_dict, conf.hdbscanclass_dict)
    fig.savefig(conf.alt_colour_plot)

    ## append final column with consolidated classification: optimal, alternative
    indata['optimal_labels'] = optimal_labels
    indata['alternative_labels'] = alternative_labels

    ## save table with labels to file
    astropy.io.ascii.write(indata, output = conf.Catalogue_with_hdbscan_labels, format = 'csv', overwrite=True)

    ## final performance
    #get true labels in terms of hdbscan classes
    true_labels = indata[f'{conf.targetname}'].copy()
    true_labels[indata[f'{conf.targetname}']==conf.hclass_dict['galaxy']] = conf.hdbscanclass_dict['galaxy'] #gal
    true_labels[indata[f'{conf.targetname}']==conf.hclass_dict['star']] = conf.hdbscanclass_dict['star'] #star
    true_labels[indata[f'{conf.targetname}']==conf.hclass_dict['qso']] = conf.hdbscanclass_dict['qso'] #qso

    #compute metric scores for optimal and alternative labels
    opt_results_str, metrics_list_opt = lib.compute_metric_scores(true_labels,optimal_labels)
    print('Metric scores for optimal labels.\n'+opt_results_str)
    fh_op.write('Metric scores for optimal labels.\n'+opt_results_str)

    alt_results_str, metrics_list_alt = lib.compute_metric_scores(true_labels,alternative_labels)
    print('Metric scores for alternative labels.\n'+alt_results_str)
    fh_op.write('Metric scores for alternative labels.\n'+alt_results_str)

    fh_op.close()

    ## plot confusion matrix for optimal and alternative labels
    y_pred_optimal = optimal_labels.copy()
    y_pred_alternative = alternative_labels.copy()
    y_true = np.array(indata[f'{conf.targetname}'].copy())

    for cons_type,y_pred in zip(['optimal','alternative'],[y_pred_optimal,y_pred_alternative]):
        #normalized
        identifiers = ['outlier','star','galaxy','qso']
        fig, ax = lib.plot_confusion_matrix(y_true, y_pred, identifiers, normalize = True)
        fig.savefig( conf.confusion_plots.replace('CLASS',f'all_normalized_consolidation_{cons_type}') )

        for identifiers in [['star'],['galaxy'],['qso']]:
            fig, ax = lib.plot_confusion_matrix(y_true, y_pred, identifiers, normalize = True)
            fig.savefig( conf.confusion_plots.replace('CLASS',f'{identifiers[0]}_normalized_consolidation_{cons_type}') )

        #not normalized
        identifiers = ['outlier','star','galaxy','qso']
        fig, ax = lib.plot_confusion_matrix(y_true, y_pred, identifiers, normalize = False)
        fig.savefig( conf.confusion_plots.replace('CLASS',f'all_consolidation_{cons_type}') )

        for identifiers in [['star'],['galaxy'],['qso']]:
            fig, ax = lib.plot_confusion_matrix(y_true, y_pred, identifiers, normalize = False)
            fig.savefig( conf.confusion_plots.replace('CLASS',f'{identifiers[0]}_consolidation_{cons_type}') )


if __name__ == '__main__':
    conf = ConfigVars(test = False)
    lib = HelperFunctions(conf)
    run_consolidation(conf, lib)
