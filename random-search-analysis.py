import pickle
import pandas as pd
import statistics
import numpy as np


# NOTE: FORMAT OF PICKLE:
# (PR_area, confusion_matrix(y_val_target, prediction).ravel(), roc_auc, parameters, time.time() - start_time)
# filters, filterSize, dropout, units (1,2), learningRate, weight = parameters


# BELOW IS FOR NEW RANDOM SEARCHES

pickle_in = open("random_search2", "rb")
random_search = pickle.load(pickle_in)


# sorting by 0th index (pr_auc)
random_search.sort(key=lambda x: x[0], reverse=True)

print(random_search)

# code for exporting distribution plot info
# pr_auc = [i[0] for i in random_search]
# result_df = pd.DataFrame({'pr_auc': pr_auc})
# result_df.to_csv("opioid_single.csv", sep=',', index=False)


# THIS IS FOR ANALYZING TRAINING W/ RANDOM SEARCH
single_df = pd.read_csv("opioid_single.csv")
multi1_df = pd.read_csv("opioid_multi1.csv")
multi1_no_weights_df = pd.read_csv("opioid_multi1_no_weights.csv")
multi2_df = pd.read_csv("opioid_multi2.csv")


# getting pr_auc
single = list(single_df['pr_auc'])
multi1 = list(multi1_df['pr_auc'])
multi1_no_weights = list(multi1_no_weights_df['pr_auc'])
multi2 = list(multi2_df['pr_auc'])

print("SINGLE")
print("Median: ", statistics.median(single))
print("Std. Deviation: ", np.std(single))
print('Max: ', max(single))
print()

print("MULTI1")
print("Median: ", statistics.median(multi1))
print("Std. Deviation: ", np.std(multi1))
print('Max: ', max(multi1))
print()

print("MULTI1 NO WEIGHTS")
print("Median: ", statistics.median(multi1_no_weights))
print("Std. Deviation: ", np.std(multi1_no_weights))
print('Max: ', max(multi1_no_weights))
print()

print("MULTI2")
print("Median: ", statistics.median(multi2))
print("Std. Deviation: ", np.std(multi2))
print('Max: ', max(multi2))
print()



# PLOTS OF DISTRIBUTIONS ARE IN VISUALIZATION

