branch = 'personalization-1.0.0'
iterations = 20
epochs = 20
maglr_iterations = 25
maglr_epochs = 25
#50 points for cipic
points_per_ring = 50
percent_test_points = .1 #0.005 #Gives single test data point for scut 
percent_valid_points = .2
#Removing equally spaced data. 10% for 
num_test_points = int(round(percent_test_points*points_per_ring,1))
num_valid_points = int(round(percent_valid_points*points_per_ring,1))   
test_seed = 0
validation_seed = 100
batch_size = 32



# System Definitions
POS_DIM=0
NUMPOINTS_DIM=1
EAR_DIM=2

LEFT_EAR_INDEX=0
RIGHT_EAR_INDEX=1

# END System Definitions


'''Model dependencies (which models depend on (->) which other models)
magri -> real, realmean, realstd, imag, imagmean, imagstd
'''
model_deps = {}
model_deps['magri'] = ['real','realmean', 'realstd', 'imag', 'imagmean', 'imagstd']
model_deps['magfinal'] = ['mag', 'magri']
model_deps['magtotal'] = ['magfinal', 'mag', 'magri', 'magl', 'magr']

'''Scut special parameters'''
SCUT_RADII = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25] #to train with all, you can also say SCUT_RADII = None
databases_with_no_anthro = ['scut']

num_ear_params = 10
num_head_params = 17

'''Specify the models to train. Options are
real, realdiff, realmean, realstd
imag, imagdiff, imagmrean, imagstd
magri

This will automatically take care of creating and training networks on which these models depend
ex: ['magri'] magri depends on all other networks. All networks will be trained
ex: ['realdiff'] realdiff depends on 'real'. Only 'real' and 'realdiff' will be trained
'''
models_to_train_1 = ['mag', 'magl', 'magr', 'real', 'imag', 'realmean', 'realstd', 'imagmean', 'imagstd'] #[] #['imag', 'imagmean', 'imagstd'] #[] #['magl', 'magr'] #
models_to_train_2 = ['magri']
models_to_train_3 = ['magfinal']
models_to_train_4 = ['magtotal']
# uncomment the line below if you want to predict other models
# models_to_predict = ['magl', 'maglmean', 'maglstd'] # [ 'magr', 'magrmean', 'magrstd'] #['mag', 'magri', 'magfinal', 'magtotal'] #

models_to_predict= [ 'magtotal'] # 'magl', 'magr'] # ,['magl', 'magr', 'maglmean', 'maglstd', 'magrmean', 'magrstd'] # 
models_to_eval = models_to_predict
models_to_renormalize = [] # ['mag', 'magri', 'magfinal']

'''For meanstd_analysis script'''
models_to_analyze = ['magl', 'magr']
models_to_analyze = ['magtotal'] 
subjects_to_analyze = ['all']

'''For analysis script'''
analyze_models = True

'''Specify the models you wish you include in the final model
ex: ['magri'] the output of magri left and magri right will be in the final network
ex: ['magri', 'realdiff'] the left and right ears of magri and realdiff networks will be included in the final network'''
finals = ['magtotal']

