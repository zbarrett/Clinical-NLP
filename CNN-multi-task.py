import numpy as np
import pickle
import gc
import random
import time
start_time = time.time()

from keras.utils import plot_model


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from keras import regularizers
from keras import optimizers
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve, auc
from keras.callbacks import Callback, EarlyStopping

from keras import Input
from keras import layers
from keras import Model

# from keras.backend.tensorflow_backend import set_session
# from keras.backend.tensorflow_backend import clear_session
# from keras.backend.tensorflow_backend import get_session



# below allow for reproducible results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os
os.environ['PYTHONHASHSEED'] = str(3)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
random.seed(4)

from keras import backend as k

""" refer to single-task CNN file for more thorough comments -- (i.e. preprocessing) """

pickle_in = open("preprocessed_opioid_data_multi", "rb")
preprocessed_opioid_data = pickle.load(pickle_in)


X_train, y_train_target, train_aux, X_val, y_val_target, val_aux, \
    x_test, y_test_target, test_aux, vocab_size, word_index = preprocessed_opioid_data

maxlen = 35000
print("maxlen:" + str(maxlen))
print("vocab_size: " + str(vocab_size))


# converting into logistic regression -- no duplicate billing codes?
# for i in range(len(train_aux)):
#     for j in range(len(train_aux[i])):
#         if train_aux[i][j] > 1:
#             train_aux[i][j] = 1
#
# for i in range(len(val_aux)):
#     for j in range(len(val_aux[i])):
#         if val_aux[i][j] > 1:
#             val_aux[i][j] = 1

# building CNN using functional API

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precision = []
        self.val_auc_rocs = []
        self.area = []

    # NEED TO REWRITE
    def on_epoch_end(self, epoch, logs={}):
        # below is for ROC AUC
        # taking 0th element
        val_predict = (np.array(self.model.predict(self.validation_data[0], batch_size=1)[0])).round()
        val_predictROC = (np.asarray(self.model.predict(self.validation_data[0], batch_size=1)[0]))
        val_true = self.validation_data[1]
        val_auc_roc = roc_auc_score(val_true, val_predictROC)

        # from here dealing with PR AUC

        """
        calculate area under precision recall curve
        note that x-axis should be recall and y-axis should be precision
        """
        val_f1 = f1_score(val_true, val_predict)
        val_recall = recall_score(val_true, val_predict)
        val_precision = precision_score(val_true, val_predict)

        precision, recall, _ = precision_recall_curve(y_true=val_true, probas_pred=val_predictROC)

        area = auc(recall, precision)
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_auc_rocs.append(val_auc_roc)
        self.area.append(area)

        print(" - ROC_AUC: %f" % (val_auc_roc))
        print(" - PR_AUC: %f" %(area))


auc_metrics = Metrics()

# randomly selects num_combos # of params
def randomSearch(num_combos):
    params = []
    while (len(params) < num_combos):
        # CNN hyperparams
        filters = random.randint(32, 512)
        filterSize = random.choice(tuple({1, 3}))
        dropout = random.uniform(0, .5)
        units1 = random.randint(32, 256)
        units2 = random.randint(16, 128)
        learningRate = random.uniform(0.00001, 0.005)

        # different keras optimizers
        optimizer = random.choice(tuple({'adam', 'rmsprop'}))

        # this is specific to MTL
        weight = random.randint(1, 30)

        value = (filters, filterSize, dropout, units1, units2, learningRate, weight, optimizer)
        params.append(value)

    return params



erstop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, restore_best_weights=True)
auc_calc = []
combinedFilters = []
bestConfig = {}

# GETTING DESIRED AUX LAYERS
def createSingleAuxLayer(i):
    name = 'aux' + str(i)
    return layers.Dense(1, name=name, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))


# dicts allow for convenient retrieval of layer names and data
aux_list = ['aux'+str(i) for i in range(len(train_aux))]
train_dict = {}
val_dict = {}
for i in range(len(aux_list)):
    name = aux_list[i]
    train_dict[name] = train_aux[i]
    val_dict[name] = val_aux[i]


# was going to be a method to create all aux layers using global keyword -- would not be super useful
# would make code more confusing to read and would only save a negligible amount of characters
# def createAuxLayers(train, val):
#     layers = []
#     for i in range(len(train_aux)):
#         name = 'aux' + str(i)
#
#         global layer_name



# collecting random search criteria
# hyperparams = randomSearch(50)  # 50

# when not performing random search
hyperparams = [(247, 3, 0.12429298701672126, 141, 122, 0.001086441692619866, 2, 'rmsprop')]


# NOTE --- STILL STRUGGLING ON GETTING REPRODUCIBLE RESULTS
hyperparam_results_PR = []
for parameters in hyperparams:
    filters, filterSize, dropout, units1, units2, learningRate, weight, optimizer = parameters  # units1

    if optimizer == 'adam':
        optimizer = optimizers.Adam(lr=learningRate)
    else:
        optimizer = optimizers.RMSprop(lr=learningRate)

    # using keras's functional API
    input_tensor = Input(shape=(maxlen,))
    hidden = layers.Embedding(input_dim=vocab_size + 1, output_dim=300, input_length=maxlen)(input_tensor)
    hidden = layers.Conv1D(filters, filterSize, activation='relu')(hidden)
    hidden = layers.GlobalMaxPooling1D()(hidden)
    # hidden = layers.Dense(units1, activation='relu')(hidden)
    hidden = layers.Dropout(dropout)(hidden)
    hidden = layers.Dense(units2, activation='relu')(hidden)
    target_output = layers.Dense(1, name='target', activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(hidden)

    # creating aux layers
    aux_output0 = createSingleAuxLayer(0)(hidden)
    aux_output1 = createSingleAuxLayer(1)(hidden)
    aux_output2 = createSingleAuxLayer(2)(hidden)
    aux_output3 = createSingleAuxLayer(3)(hidden)
    aux_output4 = createSingleAuxLayer(4)(hidden)
    aux_output5 = createSingleAuxLayer(5)(hidden)
    aux_output6 = createSingleAuxLayer(6)(hidden)
    aux_output7 = createSingleAuxLayer(7)(hidden)
    aux_output8 = createSingleAuxLayer(8)(hidden)
    aux_output9 = createSingleAuxLayer(9)(hidden)
    aux_output10 = createSingleAuxLayer(10)(hidden)
    aux_output11 = createSingleAuxLayer(11)(hidden)
    aux_output12 = createSingleAuxLayer(12)(hidden)
    aux_output13 = createSingleAuxLayer(13)(hidden)
    aux_output14 = createSingleAuxLayer(14)(hidden)
    aux_output15 = createSingleAuxLayer(15)(hidden)
    aux_output16 = createSingleAuxLayer(16)(hidden)
    aux_output17 = createSingleAuxLayer(17)(hidden)
    aux_output18 = createSingleAuxLayer(18)(hidden)

    model = Model(input_tensor, [target_output, aux_output0, aux_output1, aux_output2, aux_output3, aux_output4, aux_output5,
                                 aux_output6, aux_output7, aux_output8, aux_output9, aux_output10, aux_output11, aux_output12,
                                 aux_output13, aux_output14, aux_output15, aux_output16, aux_output17, aux_output18])
    model.summary()
    # plot_model(model, to_file='multi_1.png')


    # creating dicts for simplicity
    loss_dict = {'target': 'binary_crossentropy'}
    loss_weights_dict = {'target': weight}  # making loss pertaining to target much more important than other losses
    for i in aux_list:
        loss_dict[i] = 'binary_crossentropy'
        loss_weights_dict[i] = 1

    # make note of metric -- would a different one be better?
    model.compile(optimizer=optimizer,
                  loss=loss_dict,  # pick appropriate loss for each layer
                  loss_weights=loss_weights_dict,
                  metrics=['acc'])

    # fitting model -- note list for y's

    # adding the target data to dicts
    train_dict['target'] = np.array(y_train_target)
    val_dict['target'] = np.array(y_val_target)

    # change epochs to 30
    x = model.fit(X_train, train_dict, epochs=30, batch_size=1,
                  validation_data=(X_val, val_dict),
                  callbacks=[auc_metrics, erstop])

    y_pred = model.predict(X_val, batch_size=1)
    y_pred_target = y_pred[0]

    prediction = []
    predictProb = []
    # note -- interested in the prediction for target (i = 0)
    for x in y_pred_target:
        predictProb.append(x[0])
        if x[0] < 0.5:
            prediction.append(0)
        else:
            prediction.append(1)



    accuracy_val = accuracy_score(np.array(y_val_target), prediction)
    print("Accuracy:" + str(accuracy_val))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val_target, prediction))
    print()
    print("Classification Report:")
    print(classification_report(y_val_target, prediction))
    print()
    roc_auc = str(roc_auc_score(y_val_target, y_pred_target)) # prediction?
    print("AUC_ROC area: " + str(roc_auc))
    print()
    precision, recall, _ = precision_recall_curve(y_val_target, y_pred_target)
    PR_area = str(auc(recall, precision))
    print("PR area: " + PR_area)
    print()



# NOT READY FOR TEST YET
# print("For test Dataset")
# predictionforROC_test = model.predict(np.asarray(x_test), batch_size = 1)
#
# prediction1 = []
# predictProb1 = []

# NEED TO UPDATE THIS LIKE ABOVE CODE (i.e. 0's for predictionforROCtest & with target
# for x in predictionforROC_test: # FIXME ABOVE
#     predictProb1.append(x[0])
#     if x[0] < 0.5:
#         prediction1.append(0)
#     else:
#         prediction1.append(1)
#
# accuracy_test = accuracy_score(np.array(y_test), prediction1)
# print("Accuracy_test: " + str(accuracy_test))
# print(classification_report(y_test, prediction1))
# roc_auc1 = str(roc_auc_score(y_test, predictionforROC_test))
# print("AOC_ROC_test: " + roc_auc1)
# print()
#
#
    hyperparam_results_PR.append((PR_area, confusion_matrix(y_val_target, prediction).ravel(),
                                 roc_auc, parameters, time.time() - start_time))
    # auc_calc.append(roc_auc)
    # combinedFilters.append([filters, filterSize, dropout, units, learningRate])


    # fpr, tpr, _ = roc_curve(y_val_target, y_pred_target)
    # plotting = [precision, recall, fpr, tpr]
    # pickle_out = open("multi_plot1", "wb")
    # pickle.dump(plotting, pickle_out)
    # pickle_out.close()
    # print('PLOTTED')


    # delete model & free up memory
    del model
    gc.collect()
    if k.backend() == 'tensorflow':
        print("About to clear session")
        k.clear_session()

# FIXME
# pickle_out = open("random_search_FIXME", "wb")
# pickle.dump(hyperparam_results_PR, pickle_out)
# pickle_out.close()

# result_df = pd.DataFrame({'pr_auc': auc_calc, 'parameters': combinedFilters})
# result_df.to_csv('opioidCNN_functional.csv', sep='|', index=False)






