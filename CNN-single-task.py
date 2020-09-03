# creating initial CNN model which will be built on to create multi-task learning model

import pandas as pd
import numpy as np
import pickle
import random
import gc

from keras import backend as k
from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from keras import regularizers
from keras import optimizers
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve, auc
from keras.callbacks import Callback, EarlyStopping

# figure out which keras imports are needed

pickle_in = open("preprocessed_opioid_data", "rb")
preprocessed_opioid_data = pickle.load(pickle_in)

X_train, y_train, X_val, y_val, x_test, y_test, vocab_size, word_index = preprocessed_opioid_data

# building CNN

# class to collect metrics
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precision = []
        self.val_auc_rocs = []
        self.area = []

    def on_epoch_end(self, epoch, logs={}):
        val_predictROC = (np.asarray(self.model.predict(self.validation_data[0], batch_size=1)))
        val_true = self.validation_data[1]
        val_auc_roc = roc_auc_score(val_true, val_predictROC)
        val_predict = (np.array(self.model.predict(self.validation_data[0], batch_size=1))).round()

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
        print(" - PR_AUC: %f" % (area))

        # precision, recall, _ = precision_recall_curve(y_true=val_tar, probas_pred=val_predictROC)
        # are = auc(recall, precision)
        # self.val_f1s.append(val_f1)
        # self.val_recalls.append(val_recall)
        self.val_auc_rocs.append(val_auc_roc)
        # self.area.append(are)
        print(" - ROC_AUC: %f" % (val_auc_roc))
        # print(" - PR_AUC: %f" %(area))


auc_metrics = Metrics()


# method for random search
def randomSearch(num_combos):
    params = []
    while (len(params) < num_combos):
        # CNN hyperparams
        filters = random.randint(32, 512)
        filterSize = random.choice(tuple({1, 3}))
        dropout = random.uniform(0, .5)
        units = random.randint(32, 256)
        learningRate = random.uniform(0.00001, 0.005)

        # different keras optimizers
        optimizer = random.choice(tuple({'adam', 'rmsprop'}))

        value = (filters, filterSize, dropout, units, learningRate, optimizer)
        params.append(value)

    return params

# UNCOMMENT TO PERFORM RANDOM SEARCH
# hyperparams = randomSearch(50)
maxlen = 35000
hyperparams = [(270, 1, 0.010738845168609346, 67, 0.0008978569139794033, 'adam')]

hyperparam_results_PR = []
for params in hyperparams:

    filters, filterSize, dropout, units, learningRate, optimizer = params

    if optimizer == 'adam':
        optimizer = optimizers.Adam(lr=learningRate)
    else:
        optimizer = optimizers.RMSprop(lr=learningRate)

    print('filters: ', filters)
    print('filter size: ', filterSize)  # should always be 1
    print('dropout: ', dropout)
    print('learning rate: ', learningRate)
    print('units: ', units, '\n')

    erstop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, restore_best_weights=True)
    auc_calc = []
    combinedFilters = []
    bestConfig = {}

    model = Sequential()
    model.add(Embedding(vocab_size + 1, 300, input_length=maxlen))
    model.add(Conv1D(filters, filterSize, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    print('SUMMARY: ')
    model.summary()

    adam = optimizers.Adam(lr=learningRate)
    rmsprop = optimizers.RMSprop(lr=learningRate)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    x = model.fit(np.array(X_train), np.array(y_train), epochs=30, batch_size=1,
                  validation_data=(np.array(X_val), np.array(y_val)), callbacks=[auc_metrics, erstop])

    # predictionforROC = model.predict(np.array(X_val), batch_size = 1)
    # print(predictionforROC)

    # model.save('DAN_Word_AvgPlusMaxPool_Opioid.h5')
    # model = load_model('CNN_OpioidCUIS.h5')
    model.summary()
    print("For validation dataset")
    # prediction_val = model.predict_classes(np.array(X_val), batch_size = 1)
    predictionforROC_val = model.predict(np.asarray(X_val), batch_size=1)

    prediction = []
    predictProb = []
    for x in predictionforROC_val:
        predictProb.append(x[0])
        if x[0] < 0.5:
            prediction.append(0)
        else:
            prediction.append(1)
    # print(y_test)
    # print(prediction)

    accuracy_val = accuracy_score(np.array(y_val), prediction)
    print("Accuracy:" + str(accuracy_val))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, prediction))
    print()
    print("Classification Report:")
    print(classification_report(y_val, prediction))
    print()
    roc_auc = str(roc_auc_score(y_val, predictionforROC_val))
    print("AUC_ROC area: " + roc_auc)
    print()
    precision, recall, _ = precision_recall_curve(y_val, predictionforROC_val)
    PR_area = str(auc(recall, precision))
    print("PR area: " + PR_area)
    print()

    # text_sample = X_val[1]
    # c = make_pipeline(tokenizer
    # print(test_sample)

    # explainer = LimeTextExplainer()
    # print(X_val[0])
    # explanation = explainer.explain_instance(X_val[1], model.predict(batch_size = 1), num_features = 6, top_labels = 5)
    # print("Reached here")
    # weights = OrderedDict(explanation.as_list())
    # lime_weight = pd.DataFrame({'words': list(weights.keys()), 'weights': list(weights.values())})
    # print(lime_weight.head(20))

    # prob_df = pd.DataFrame({'encounter_ID':encounterList, 'prob':predictProb, 'predict_label': prediction, 'gold_label': y_val})
    # print(prob_df.head(10))
    # prob_df.to_csv("DAN_Word_AvgPool_Opioid_Test.csv", sep = ',', index = False)
    """
    print("For test Dataset")
    predictionforROC_test = model.predict(np.asarray(x_test), batch_size = 1)
    
    prediction1 = []
    predictProb1 = []
    for x in predictionforROC_test:
        predictProb1.append(x[0])
        if x[0] < 0.5:
            prediction1.append(0)
        else:
            prediction1.append(1)
    
    accuracy_test = accuracy_score(np.array(y_test), prediction1)
    print("Accuracy_test: " + str(accuracy_test))
    print(classification_report(y_test, prediction1))
    roc_auc1 = str(roc_auc_score(y_test, predictionforROC_test))
    print("AOC_ROC_test: " + roc_auc1)
    print()
    """
    # prob_df1 = pd.DataFrame({'encounterID': encounterList, 'prob': predictProb1, 'predict_label': prediction1, 'gold_label': y_test})
    # prob_df1.to_csv("DAV_MAXPool_test_predict.csv", sep = ',', index = False)

    # bestConfig[roc_auc] = [filters, filterSize, dropout, learningRate]

    hyperparam_results_PR.append((PR_area, confusion_matrix(y_val, prediction).ravel(),
                                  roc_auc, params))

    del model
    gc.collect()
    if k.backend() == 'tensorflow':
        print("About to clear session")
        k.clear_session()



# pickle_out = open("random_search_single", "wb")
# pickle.dump(hyperparam_results_PR, pickle_out)
# pickle_out.close()


result_df = pd.DataFrame({'roc_auc': auc, 'parameters': combinedFilters})
result_df.to_csv('opioidCNN.csv', sep='|', index=False)

fpr, tpr, _ = roc_curve(y_val, predictionforROC_val)

# plotting = [precision, recall, fpr, tpr]
# pickle_out = open("single_task_plot", "wb")
# pickle.dump(plotting, pickle_out)
# pickle_out.close()

