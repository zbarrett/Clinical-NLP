import os
import pandas as pd
import numpy as np
import pickle
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer


# helper method for reading files
def openFilesInDir(item, path):
    full_path = path + str(item) + ".txt"
    if (os.path.isfile(full_path)):
        return full_path
    else:
        return "No"

# helper method to process data
def processText(train_data, val_data, test_data, maxlen):
    tokenizer = Tokenizer(oov_token='UNK', lower=False)  # tokenizes words, removes punctuations
    # oov -- out of vocab, if something is oov, inputs UNK as a placeholder (index 0)

    # extracting fields
    y_train = train_data.gold_label
    x_train = train_data.notes
    y_val = val_data.gold_label
    x_val = val_data.notes
    y_test = test_data.gold_label
    x_test = test_data.notes

    # pickling tokenizer
    # tokenizer.fit_on_texts(x_train)
    # with open('CUIS_tokenizer.pkl', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # unpickling
    with open('CUIS_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # dict of mappings
    word_index = tokenizer.word_index
    a = len(word_index)
    print("number of words:" + str(a) + '\n')
    print("fit on X_train completed")

    sequences = tokenizer.texts_to_sequences(x_train)  # converts to list of tokens (cuis)
    print("Text to sequence completed")

    x_train = pad_sequences(sequences, maxlen=maxlen)  # ensures that each sequence has same length, adds 0s to do so
    print("X_train pad_sequences completed")
    print("X_train: ", x_train)

    sequences_val = tokenizer.texts_to_sequences(x_val)
    x_val = pad_sequences(sequences_val, maxlen=maxlen)
    print("X_Val pad_sequences completed")

    sequences_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(sequences_test, maxlen=maxlen)
    print("X_test pad_sequences completed \n")

    return x_train, y_train, x_val, y_val, x_test, y_test, a, word_index


def processTextMulti(train_data, val_data, test_data, maxlen):
    tokenizer = Tokenizer(oov_token='UNK', lower=False)  # tokenizes words, removes punctuations
    # oov -- out of vocab, if something is oov, inputs UNK as a placeholder (index 0)

    # extracting fields
    y_train = list(zip(train_data.gold_label, train_data.icd9, train_data.icd10))
    x_train = train_data.notes
    y_val = list(zip(val_data.gold_label, val_data.icd9, val_data.icd10))
    x_val = val_data.notes
    y_test = list(zip(test_data.gold_label, test_data.icd9, test_data.icd10))
    x_test = test_data.notes

    # pickling tokenizer
    # tokenizer.fit_on_texts(x_train)
    # with open('CUIS_tokenizer.pkl', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # unpickling
    with open('CUIS_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # dict of mappings
    word_index = tokenizer.word_index
    a = len(word_index)
    print("number of words:" + str(a) + '\n')
    print("fit on X_train completed")

    sequences = tokenizer.texts_to_sequences(x_train)  # converts to list of tokens (cuis)
    print("Text to sequence completed")

    x_train = pad_sequences(sequences, maxlen=maxlen)  # ensures that each sequence has same length, adds 0s to do so
    print("X_train pad_sequences completed")
    print("X_train: ", x_train)

    sequences_val = tokenizer.texts_to_sequences(x_val)
    x_val = pad_sequences(sequences_val, maxlen=maxlen)
    print("X_Val pad_sequences completed")

    sequences_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(sequences_test, maxlen=maxlen)
    print("X_test pad_sequences completed \n")

    return x_train, y_train, x_val, y_val, x_test, y_test, a, word_index


""" UNCOMMENT TO CREATE THE DATA FOR SINGLE TASK LEARNING

# OpioidAnnotated_DataSplit.csv contains IDs & split & label
# split: 1 = train, 2 = val, 3 = test

# load labels
data_df = pd.read_csv('/Archive-Odin/Opioid_annotated/OpioidAnnotated_DataSplit.csv', sep='|')
print('labels head: \n', data_df.head(), '\n')

# shuffle df
data_df = data_df.sample(frac=1, random_state=12).reset_index(drop=True)

# prepare to add in NCUIs
data_df['notes'] = data_df['hsp_account_id']
# NOTE: NCUIS contains all CUIS + negation of some CUIS
# (i.e. same notes in Data_CUIS and Data_NCUIS)
path = '/Archive-Odin/Opioid_Annotated_Encounters/Data_NCUIS/'


# collect and read files, insert in notes field
data_df['notes'] = data_df.notes.apply(lambda x: openFilesInDir(x, path))  # list of files
print('files head: \n', data_df.head(), '\n')
data_df['notes'] = data_df.notes.apply(lambda x: open(x, "r").read())  # opens files
print('read files head: \n', data_df.head(), '\n')

patient_train = data_df[data_df.splitType == 1].reset_index(drop=True)
patient_val = data_df[data_df.splitType == 2].reset_index(drop=True)
patient_test = data_df[data_df.splitType == 3].reset_index(drop=True)


# setting max length of CUIs
maxlen = 35000

# splitting and tokenizing
X_train, y_train, X_val, y_val, x_test, y_test, vocab_size, word_index = \
    processText(patient_train, patient_val, patient_test, maxlen)

print("maxlen:" + str(maxlen))
print("vocab_size: " + str(vocab_size))

# creating a list to pickle the preprocessed information
# preprocessed_opioid_data = [X_train, y_train, X_val, y_val, x_test, y_test, vocab_size, word_index]
#
# pickle_out = open("preprocessed_opioid_data", "wb")
# pickle.dump(preprocessed_opioid_data, pickle_out)
# pickle_out.close()

"""




"""
# MULTI PREPROCESSING
# this code is same as above
data_df = pd.read_csv('/Archive-Odin/Opioid_annotated/OpioidAnnotated_DataSplit.csv', sep='|')
print('labels head: \n', data_df.head(), '\n')

# shuffle df
data_df = data_df.sample(frac=1, random_state=12).reset_index(drop=True)

# prepare to add in NCUIs
data_df['notes'] = data_df['hsp_account_id']
# NOTE: NCUIS contains all CUIS + negation of some CUIS
# (i.e. same notes in Data_CUIS and Data_NCUIS)
path = '/Archive-Odin/Opioid_Annotated_Encounters/Data_NCUIS/'


# collect and read files, insert in notes field
data_df['notes'] = data_df.notes.apply(lambda x: openFilesInDir(x, path))  # list of files
print('files head: \n', data_df.head(), '\n')
data_df['notes'] = data_df.notes.apply(lambda x: open(x, "r").read())  # opens files
print('read files head: \n', data_df.head(), '\n')

# below code is for processing billing codes
billing_data_df = pd.read_csv('./billing_data/uc_dx.txt', sep='|')

print('billing_data_df head: ', billing_data_df)
print('billing labels: ', list(billing_data_df.columns.values))

# remove rows not pertaining to data
def reduceBillingCodes(id_data, billing):
    ids = list(id_data['hsp_account_id'])
    drop_list = []
    for i in range(len(billing['hsp_account_id'])):
        id = billing['hsp_account_id'].iloc[i]
        if id not in ids:
            drop_list.append(i)

    billing = billing.drop(drop_list)

    return billing


# reducing billing codes -- takes forever to run (pickled)
# billing_data_df = reduceBillingCodes(data_df, billing_data_df)

# pickle_out = open("relevant_billing_raw", "wb")
# pickle.dump(billing_data_df, pickle_out)
# pickle_out.close()

pickle_in = open("relevant_billing_raw", "rb")
billing_data_df = pickle.load(pickle_in)


# splitting into ICD 9 & 10 codes
billing_data_df_icd9 = billing_data_df[billing_data_df['icd9_or_icd10'] == 9]
billing_data_df_icd10 = billing_data_df[billing_data_df['icd9_or_icd10'] == 10]

# collapsing df by id -- groupby has really weird behavior & drops "nuissance" columns
billing_data_df_icd9 = billing_data_df_icd9.groupby(['hsp_account_id'], as_index=False).agg(lambda x: ' '.join(x))  # x.unique()
billing_data_df_icd10 = billing_data_df_icd10.groupby(['hsp_account_id'], as_index=False).agg(lambda x: ' '.join(x))  # x.unique()


print ('labels: ', list(billing_data_df.columns.values))


# method to match by id
def matchID(id, billing):
    for i in range(len(billing['hsp_account_id'])):
        billing_id = billing['hsp_account_id'].iat[i]
        if id == billing_id:
            return billing['dx_code'].iat[i]

    return ''


# method to match labels of df
data_df['icd9'] = data_df['hsp_account_id'].apply(lambda x: matchID(id=x, billing=billing_data_df_icd9))
data_df['icd10'] = data_df['hsp_account_id'].apply(lambda x: matchID(id=x, billing=billing_data_df_icd10))



pickle_out = open("data_df_with_billing_codes", "wb")
pickle.dump(data_df, pickle_out)
pickle_out.close()

"""

# now splitting appropriately
pickle_in = open("data_df_with_billing_codes", "rb")
data_df = pickle.load(pickle_in)

patient_train = data_df[data_df.splitType == 1].reset_index(drop=True)
patient_val = data_df[data_df.splitType == 2].reset_index(drop=True)
patient_test = data_df[data_df.splitType == 3].reset_index(drop=True)


# setting max length of CUIs
maxlen = 35000

# splitting and tokenizing
X_train, y_train, X_val, y_val, x_test, y_test, vocab_size, word_index = \
    processTextMulti(patient_train, patient_val, patient_test, maxlen)


# parsing ICDs appropriately

# y[0] -- label, y[1] -- icd9, y[2] -- icd10

# INCOMPLETE METHOD -- returns the pruned billing codes (not targeted at opioid misuse)
# DO NOT RUN IF TRYING TO GET OPIOID BILLING CODES
"""
def multiLabelParserBulk(y):
    y_target = [i[0] for i in y]
    # reg exp to reduce dimensionality -- GO BACK TO SLACK OPIOIDS 3/27
    y_aux = [(i[1], i[2]) for i in y]
    # y_aux = [re.sub("[a-zA-Z]", "", i[1]) for i in y]
    # y_aux = [re.sub("\.[0-9]*", "", i) for i in y_aux]

    # cv = CountVectorizer()
    # y_aux = cv.fit_transform(y_aux).toarray()
    y_target = y_target

    return (y_target, y_aux)


# y_train_target, y_train_aux = multiLabelParserBulk(y_train)

"""

# creating list of opioid-specific billing codes
icd9 = ['30550', '30551', '30552', '30400', '30401', '30402', '30470', '30471', '30472', '96500', '96501', '96502',
        '96509', 'E8500']

icd10 = ['F1110', 'F11120', 'F11121', 'F11122', 'F11129', 'F1114', 'F11150', 'F11151', 'F11159', 'F11181', 'F11182',
         'F11188', 'F1120', 'F11220', 'F11221', 'F11222', 'F11229', 'F1123', 'F1124', 'F11250', 'F11251', 'F11259',
         'F11281', 'F11282', 'F11288', 'F1129', 'F1119', 'T400X4A', 'T400X4D', 'T400X4S', 'T401X4A', 'T401X4D',
         'T401X4S', 'T402X4A', 'T402X4D', 'T402X4S', 'T403X4A', 'T403X4D', 'T403X4S', 'T404X4A', 'T404X4D', 'T404X4S',
         'T40604A', 'T40604D', 'T40604S', 'T40694A', 'T40694D', 'T40694S', 'F1190', 'F11920', 'F11921', 'F11922',
         'F11929', 'F1193', 'F1194', 'F11950', 'F11951', 'F11959', 'F11981', 'F11982', 'F11988', 'F1199']


# helper method for below function
def matchICD(y, index, icd_list):
    # remove decimals
    y_icd = [re.sub("\.", "", i[index]).split() for i in y]

    # match each code to the list of opioid specific codes
    for i in range(len(y_icd)):
        entry = y_icd[i]
        relevant_codes = []
        for code in entry:
            if code in icd_list:
                relevant_codes.append(code)
        y_icd[i] = relevant_codes

    for i in range(len(y_icd)):
        y_icd[i] = " ".join(y_icd[i])

    return y_icd


# uses opioid-specific billing codes
# extracts the target and auxiliary labels (and separates ICD9/ICD10)
def multiLabelParserTarget(y):
    y_target = [i[0] for i in y]
    # matching ICD codes to opioid-specific codes
    # icd9 and icd10 refer to the lists of opioid-specific codes in the outer scope
    y_icd9 = matchICD(y, 1, icd9)
    y_icd10 = matchICD(y, 2, icd10)


    cv_icd9 = CountVectorizer()
    y_aux9 = cv_icd9.fit_transform(y_icd9).toarray()
    y_aux9_labels = cv_icd9.get_feature_names()

    cv_icd10 = CountVectorizer()
    y_aux10 = cv_icd10.fit_transform(y_icd10).toarray()
    y_aux10_labels = cv_icd10.get_feature_names()

    # NOTE: ICD9 and ICD10 codes should be looked at differently, uncomment below code to combine them anyways
    # y_aux = [0] * len(y_icd9)
    # for i in range(len(y_icd9)):
    #     y_aux[i] = str(y_icd9[i] + ' ' + y_icd10[i])
    #
    # cv = CountVectorizer()
    # y_aux = cv.fit_transform(y_aux).toarray()
    # y_aux_labels = cv.get_feature_names()

    return y_target, y_aux9, y_aux9_labels, y_aux10, y_aux10_labels


y_train_target, y_train_aux9, y_train_aux9_labels, y_train_aux10, y_train_aux10_labels = multiLabelParserTarget(y_train)
y_val_target, y_val_aux9, y_val_aux9_labels, y_val_aux10, y_val_aux10_labels = multiLabelParserTarget(y_val)
y_test_target, y_test_aux9, y_test_aux9_labels, y_test_aux10, y_test_aux10_labels = multiLabelParserTarget(y_test)


# adds empty columns for labels that are not in both sets -- not looking at test yet
def addEmptyAux(train_aux_labels, val_aux, val_aux_labels, test_aux, test_aux_labels):
    absent_labels_val = []
    absent_labels_test = []
    for i in range(len(train_aux_labels)):
        train_label = train_aux_labels[i]
        if train_label not in val_aux_labels:
            absent_labels_val.append(train_label)
        if train_label not in test_aux_labels:
            absent_labels_test.append(train_label)

    # removing labels that are not in val -- want to keep as much data, add empty columns instead of removing
    # train_aux = np.delete(train_aux, absent_labels, 1)
    # train_aux_labels = np.delete(train_aux_labels, absent_labels)


    # adding empty columns
    val_aux_labels += absent_labels_val
    empty_cols_val = np.zeros((val_aux.shape[0], len(absent_labels_val)))
    val_aux = np.append(val_aux, empty_cols_val, axis=1)

    test_aux_labels += absent_labels_test
    empty_cols_test = np.zeros((test_aux.shape[0], len(absent_labels_test)))
    test_aux = np.append(test_aux, empty_cols_test, axis=1)
    print()

    return val_aux, val_aux_labels, test_aux, test_aux_labels


y_val_aux9, y_val_aux9_labels, y_test_aux9, y_test_aux9_labels = \
    addEmptyAux(y_train_aux9_labels, y_val_aux9, y_val_aux9_labels, y_test_aux9, y_test_aux9_labels)
y_val_aux10, y_val_aux10_labels, y_test_aux10, y_test_aux10_labels = \
    addEmptyAux(y_train_aux10_labels, y_val_aux10, y_val_aux10_labels, y_test_aux10, y_test_aux10_labels)


# matches aux label indices across different sets
# i.e. each column in train/val for any billing code must have the same indices
def auxIndices(train_aux_labels, val_aux_labels, test_aux_labels):
    aux_label_indices = []  # add test?
    for i in range(len(train_aux_labels)):
        train_label = train_aux_labels[i]  # add test?
        aux_label_indices.append((i, val_aux_labels.index(train_label), test_aux_labels.index(train_label)))
    return aux_label_indices


aux_label_indices9 = auxIndices(y_train_aux9_labels, y_val_aux9_labels, y_test_aux9_labels)
aux_label_indices10 = auxIndices(y_train_aux10_labels, y_val_aux10_labels, y_test_aux10_labels)

# matching up aux layers correctly
def matchAux(train_aux, val_aux, test_aux, index_list):
    train = []
    val = []
    test = []
    for i in index_list:
        temp_train = train_aux[:, i[0]]
        temp_val = val_aux[:, i[1]]
        temp_test = test_aux[:, i[2]]

        train.append(temp_train)
        val.append(temp_val)
        test.append(temp_test)


    return train, val, test


y_train_aux9, y_val_aux9, y_test_aux9 = matchAux(y_train_aux9, y_val_aux9, y_test_aux9, aux_label_indices9)
y_train_aux10, y_val_aux10, y_test_aux10 = matchAux(y_train_aux10, y_val_aux10, y_test_aux10, aux_label_indices10)


# exporting ICD codes for data quality analysis
ICDs = [y_train_target, y_train_aux9, y_train_aux9_labels, y_train_aux10, y_train_aux10_labels,
        y_val_target, y_val_aux9, y_val_aux9_labels, y_val_aux10, y_val_aux10_labels]
# pickle_out = open("aux_labels_ICDs", "wb")
# pickle.dump(ICDs, pickle_out)
# pickle_out.close()

# join ICDs back for aux label outputs
y_train_aux = y_train_aux9 + y_train_aux10
y_val_aux = y_val_aux9 + y_val_aux10
y_test_aux = y_test_aux9 + y_test_aux10


# preparing for pickle
preprocessed_opioid_data_multi = [X_train, y_train_target, y_train_aux, X_val, y_val_target, y_val_aux,
                                  x_test, y_test_target, y_test_aux, vocab_size, word_index]


# pickle_out = open("preprocessed_opioid_data_multi", "wb")
# pickle.dump(preprocessed_opioid_data_multi, pickle_out)
# pickle_out.close()




