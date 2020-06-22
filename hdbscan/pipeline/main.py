from config import ConfigVars
from helper_functions import HelperFunctions
from binary_classifier import run_binary
from classifier_consolidation import run_consolidation
from predict import run_predict

gridsearch = True
test_bool = False#True

if test_bool == True:
    gridsearch = True

print('Running star binary')
conf = ConfigVars(test = test_bool, kind = 'star', gridsearch = gridsearch)
lib = HelperFunctions(conf)
run_binary(conf, lib)

print('Running galaxy binary')
conf = ConfigVars(test = test_bool, kind = 'galaxy', gridsearch = gridsearch)
lib = HelperFunctions(conf)
run_binary(conf, lib)

print('Running qso binary')
conf = ConfigVars(test = test_bool, kind = 'qso', gridsearch = gridsearch)
lib = HelperFunctions(conf)
run_binary(conf, lib)

print('Running consolidation')
conf = ConfigVars(test = test_bool)
lib = HelperFunctions(conf)
run_consolidation(conf, lib)

print('Running prediction')
after_opt_str = run_predict(conf, lib)

if test_bool == True:
    assert after_opt_str == 'optimal consolidation method: star:     21135,     gal: 24111,     qso: 1173,     outlier: 3581.\n' , 'Test run failed'
    if after_opt_str == 'optimal consolidation method: star:     21135,     gal: 24111,     qso: 1173,     outlier: 3581.\n':
        print('Test run passed.')
