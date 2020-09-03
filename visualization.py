import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import pandas as pd


# THIS IS FOR ANALYZING TRAINING W/ RANDOM SEARCH
# single_df = pd.read_csv("opioid_single.csv")
multi1_df = pd.read_csv("opioid_multi1.csv")
multi1_no_weights_df = pd.read_csv("opioid_multi1_no_weights.csv")
multi2_df = pd.read_csv("opioid_multi2.csv")


# getting pr_auc
# single = list(single_df['pr_auc'])
multi1 = list(multi1_df['pr_auc'])
multi1_no_weights = list(multi1_no_weights_df['pr_auc'])
multi2 = list(multi2_df['pr_auc'])


"""
# HISTOGRAMS -- not super informative

# only want 50 random examples to match other searches
# multi1 = random.sample(multi1, k=50)
# multi2 = random.sample(multi2, k=50)

def bins(x):
    return x*.02


bin_breaks = [bins(x) for x in range(50)]

# plt.hist(single, bin_breaks, color='green', label='single', alpha=.25)
# plt.xlabel('PR_AUC')
# plt.ylabel('Count')
# plt.title('Single')
# plt.show()
plt.hist(multi1, bin_breaks, color='blue', label='multi (1 dense)', alpha=.25)
plt.title('Multi, 1 dense layer')
plt.xlabel('PR_AUC')
plt.ylabel('Count')
plt.show()
plt.hist(multi1_no_weights, bin_breaks, color='red', label='multi (1 dense)', alpha=.25)
plt.title('Multi, 1 dense layer, no weights')
plt.xlabel('PR_AUC')
plt.ylabel('Count')
plt.show()
plt.hist(multi2, bin_breaks, color='purple', label='multi (2 dense)', alpha=.25)
plt.title('Multi, 2 dense layers')
plt.xlabel('PR_AUC')
plt.ylabel('Count')
plt.show()


# plt.hist(single, bin_breaks, color='green', label='single', alpha=.25)
# plt.hist(multi1, bin_breaks, color='blue', label='weights', alpha=.25)
# plt.hist(multi1_no_weights, bin_breaks, color='red', label='no weights', alpha=.25)
# plt.hist(multi2, bin_breaks, color='purple', label='multi (2 dense)', alpha=.25)
# plt.title('PR_AUC')
# plt.xlabel('PR_AUC')
# plt.ylabel('Count')
# plt.legend()
# plt.show()
"""


"""
# BOXPLOTS
# data_to_plot = [single, multi1, multi1_no_weights, multi2]
data_to_plot = [multi1, multi1_no_weights]

fig = plt.figure(1, figsize=(9, 6))
# Create an axes instance
ax = fig.add_subplot(111)
# Create the boxplot
bp = ax.boxplot(data_to_plot)
# ax.set_xticklabels(['Single', 'Multi, 1 dense layers', 'Multi, 2 dense layers'])
ax.set_xticklabels(['Weights', 'No Weights'])  # FIXME
ax.set_ybound(.6, .95)
# plt.title('PR_AUC')
# plt.xlabel('PR_AUC')
# plt.ylabel('Count')
# plt.legend()
plt.show()
"""





# PR_AUC & ROC_AUC
pickle_in = open("single_task_plot", "rb")
single = pickle.load(pickle_in)

pickle_in = open("multi_plot1", "rb")
multi_1 = pickle.load(pickle_in)

pickle_in = open("multi_plot2", "rb")
multi_2 = pickle.load(pickle_in)


# PR plot
plt.plot(single[1], single[0], color='red', label='Single')
plt.plot(multi_1[1], multi_1[0], color='blue', label='Multi (1)')
plt.plot(multi_2[1], multi_2[0], color='purple', label='Multi (2)')


# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend()

# show the plot
plt.show()


# ROC Curve
plt.plot(single[2], single[3], color='red', label='Single')
plt.plot(multi_1[2], multi_1[3], color='blue', label='Multi (1)')
plt.plot(multi_2[2], multi_2[3], color='purple', label='Multi (2)')


# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# show the plot
plt.show()





