from config import ConfigVars
from helper_functions import HelperFunctions
import numpy as np

def run_binary(conf, lib):
    '''
    Wrapper function for binary classifier code.

    Args:
        conf (object): contains config variables

        lib (object): contains helper functions

    Returns:

    '''''
    # Read in data
    indata = lib.get_data(conf.data_file, 'csv')

    # Select class
    target = conf.hclass_dict[conf.kind.lower()]

    # construct features, subtract (mag) or divide (fluxes)
    attribute_list, attribute_names, attribute_target = lib.get_all_features(
        indata, conf.photo_band_list, conf.combine_type, conf.targetname, target)

    ## Pre-processing
    # whitening & normalization of data
    scaled_data = lib.do_scale(np.transpose(np.array(attribute_list)), save = True)

    ## random forest classification for selection of most informative features
    ## once the RF is trained, the most important features are saved in a file
    ## PCA reads from that file, no need to re-run the RF
    if conf.gridsearch == True:
        clf = lib.do_random_forest(scaled_data, attribute_target,
                           attribute_names, kind = conf.kind, gridsearch=True)
    print('gridsearch HDBSCAN')
    ## PCA and RF top attribute selection are part of the hdbscan gridsearch
    lib.do_hdbscan_gridsearch(attribute_names=attribute_names,
                                scaled_data=scaled_data, kind = conf.kind)

    ## read the gridsearch file and identify run with best classification
    ## for a binary classifier we only care to keep the optimum e.g. star (or gal
    ## or qso) class, all other clusters are aggregated into one

    ## create performance dat files
    lib.compute_performances(indata, kind = conf.kind)

    ## write best label setups to dat file in a now binary form
    best_name, best_cluster = lib.find_best_hdbscan_setup(conf.HDBSCAN_gridsearch_performance.replace('CLASS',conf.kind))
    lib.write_best_labels_binary(best_name, best_cluster, kind = conf.kind)

    ## now write best hdbscan setup to text file
    fh = open(conf.hdbscan_best_setups_file.replace('CLASS',conf.kind), "w")
    fh.write(f"Number of top RF important attributes chosen: {best_name.split('_')[0]}\nReduced to {best_name.split('_')[1]} dimensions by PCA\nHDBSCAN min_cluster_size set to {best_name.split('_')[2]}\n")
    fh.close()

    ## using best setup, now train the binary classifier and saved trained model
    ## (for use in e.g. prediction script)
    lib.train_and_save_hdbscan(attribute_names, scaled_data, best_name, kind = conf.kind)


if __name__ == '__main__':
    gridsearch = True
    conf = ConfigVars(test = False, kind = 'star', gridsearch = gridsearch)
    lib = HelperFunctions(conf)
    run_binary(conf, lib)

    conf = ConfigVars(test = False, kind = 'galaxy', gridsearch = gridsearch)
    lib = HelperFunctions(conf)
    run_binary(conf, lib)

    conf = ConfigVars(test = False, kind = 'qso', gridsearch = gridsearch)
    lib = HelperFunctions(conf)
    run_binary(conf, lib)
