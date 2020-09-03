import numpy as np
import pickle
import matplotlib.pyplot as plt
import statistics

pickle_in = open("aux_labels_ICDs", "rb")
ICDs = pickle.load(pickle_in)

# all of these are lists
y_train_target, y_train_aux9, y_train_aux9_labels, y_train_aux10, y_train_aux10_labels, \
    y_val_target, y_val_aux9, y_val_aux9_labels, y_val_aux10, y_val_aux10_labels = ICDs

# disregard val_labels -- everything reflects train_labels

# converting data fields to np arrays
y_train_target = np.array(y_train_target)
y_train_aux9 = np.array(y_train_aux9).transpose()
y_train_aux10 = np.array(y_train_aux10).transpose()

y_val_target = np.array(y_val_target)
y_val_aux9 = np.array(y_val_aux9).transpose()
y_val_aux10 = np.array(y_val_aux10).transpose()


# combining train and val
aux9 = np.append(y_train_aux9, y_val_aux9, axis=0)
aux10 = np.append(y_train_aux10, y_val_aux10, axis=0)

TOTAL_LENGTH = aux9.shape[0]

target = np.append(y_train_target, y_val_target)
target = np.reshape(target, (TOTAL_LENGTH, 1))


# adding target label
aux9_label = np.append(aux9, target, axis=1)
aux10_label = np.append(aux10, target, axis=1)


# split by label
# this is better to be performed with lists because they can dynamically be added to
# np.append creates new array each iteration
def split(aux):
    pos = []
    neg = []

    label_index = int(aux.shape[1]) - 1

    for row in aux:
        label_val = int(row[label_index])
        if label_val == 1:
            pos.append(row)
        else:
            neg.append(row)

    pos = np.array(pos)
    neg = np.array(neg)

    return pos, neg


aux9_pos, aux9_neg = split(aux9_label)
aux10_pos, aux10_neg = split(aux10_label)


def collectCounts(aux):
    label_index = int(aux.shape[1]) - 1

    countPerPatient = np.sum(aux, axis=1)
    countPerCode = np.sum(aux, axis=0)
    countPerCode = np.delete(countPerCode, label_index)
    return [countPerPatient, countPerCode]

pos_counts9 = collectCounts(aux9_pos)
neg_counts9 = collectCounts(aux9_neg)
tot_counts9 = collectCounts(aux9_label)

pos_counts10 = collectCounts(aux10_pos)
neg_counts10 = collectCounts(aux10_neg)
tot_counts10 = collectCounts(aux10_label)


# FOR EASY READING
pos_counts9_readable = list(zip(y_train_aux9_labels, pos_counts9[1]))
neg_counts9_readable = list(zip(y_train_aux9_labels, neg_counts9[1]))
tot_counts9_readable = list(zip(y_train_aux9_labels, tot_counts9[1]))

pos_counts10_readable = list(zip(y_train_aux10_labels, pos_counts10[1]))
neg_counts10_readable = list(zip(y_train_aux10_labels, neg_counts10[1]))
tot_counts10_readable = list(zip(y_train_aux10_labels, tot_counts10[1]))


# below gathers stats about the counts
def desiredStats(count):
    results = []
    medianPerPatient = np.median(count[0])
    meanPerPatient = np.mean(count[0])
    stdPerPatient = np.std(count[0])

    medianPerCode = np.median(count[1])
    meanPerCode = np.mean(count[1])
    stdPerCode = np.std(count[1])

    results = [medianPerPatient, meanPerPatient, stdPerPatient, medianPerCode, meanPerCode, stdPerCode]

    return results


pos_stats9 = desiredStats(pos_counts9)
neg_stats9 = desiredStats(neg_counts9)
tot_stats9 = desiredStats(tot_counts9)

print('pos stats9:', pos_stats9)
print('neg stats9:', neg_stats9)
print('tot stats9:', tot_stats9)
print()

pos_stats10 = desiredStats(pos_counts10)
neg_stats10 = desiredStats(neg_counts10)
tot_stats10 = desiredStats(tot_counts10)

print('pos stats10:', pos_stats10)
print('neg stats10:', neg_stats10)
print('tot stats10:', tot_stats10)
print()
print()



# returns top n counts
def findTopN(counts, n):
    max_counts = []
    indices = []
    counts_copy = counts.copy()
    while len(indices) < n:
        max = -1
        for i in range(len(counts_copy)):
            tempCount = counts_copy[i]
            if tempCount > max:
                max = tempCount
                index = i
        max_counts.append(max)
        indices.append(index)
        counts_copy[index] = -1

    return max_counts, indices


pos9_top, pos9_indices = findTopN(pos_counts9[1], 5)
neg9_top, neg9_indices = findTopN(neg_counts9[1], 5)

pos10_top, pos10_indices = findTopN(pos_counts10[1], 5)
neg10_top, neg10_indices = findTopN(neg_counts10[1], 5)


# gets complementary values to top and the corresponding labels
def getComplement(top_indices, comp, label_names):
    complement = []
    labels = []
    for i in top_indices:
        complement.append(comp[i])
        labels.append(str(label_names[i]))

    return complement, labels


"""
# BEGINNING PLOTS

# Note: these values are for making a grouped bar graph (look at matplotlib docs)
width = .25
plt.style.use('dark_background')  # DARK MODE!! <3

# ICD 9

# plotting top pos9 & complement (i.e. corresponding negative)
# note: y_train labels are reflective of train and val
pos9_comp, pos9_names = getComplement(pos9_indices, neg_counts9[1], y_train_aux9_labels)
index = np.arange(len(pos9_names))

fig, ax = plt.subplots()
tempPos = ax.bar(index - width/2, pos9_top, width, label='Pos', color='blue')
tempNeg = ax.bar(index + width/2, pos9_comp, width, label='Neg', color='red')

ax.set_title('Top Billing Codes Positive (ICD-9)')
ax.set_xticks(index)
ax.set_xticklabels(pos9_names)
ax.legend()
plt.show()


# plotting top neg9 & complement (i.e. corresponding positive)
neg9_comp, neg9_names = getComplement(neg9_indices, pos_counts9[1], y_train_aux9_labels)
index = np.arange(len(neg9_names))

fig, ax = plt.subplots()
tempNeg = ax.bar(index - width/2, neg9_top, width, label='Neg', color ='red')
tempPos = ax.bar(index + width/2, neg9_comp, width, label='Pos', color='blue')

ax.set_title('Top Billing Codes Negative (ICD-9)')
ax.set_xticks(index)
ax.set_xticklabels(neg9_names)
ax.legend()
plt.show()


# ICD 10

# plotting top pos10 & complement (i.e. corresponding negative)
# note: y_train labels are reflective of train and val
pos10_comp, pos10_names = getComplement(pos10_indices, neg_counts10[1], y_train_aux10_labels)
index = np.arange(len(pos10_names))

fig, ax = plt.subplots()
tempPos = ax.bar(index - width/2, pos10_top, width, label='Pos', color='blue')
tempNeg = ax.bar(index + width/2, pos10_comp, width, label='Neg', color='red')

ax.set_title('Top Billing Codes Positive (ICD-10)')
ax.set_xticks(index)
ax.set_xticklabels(pos10_names)
ax.legend()
plt.show()


# plotting top neg10 & complement (i.e. corresponding positive)
neg10_comp, neg10_names = getComplement(neg10_indices, pos_counts10[1], y_train_aux10_labels)
index = np.arange(len(neg10_names))

fig, ax = plt.subplots()
tempNeg = ax.bar(index - width/2, neg10_top, width, label='Neg', color ='red')
tempPos = ax.bar(index + width/2, neg10_comp, width, label='Pos', color='blue')

ax.set_title('Top Billing Codes Negative (ICD-10)')
ax.set_xticks(index)
ax.set_xticklabels(neg10_names)
ax.legend()
plt.show()

"""


