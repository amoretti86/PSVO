import pickle

import matplotlib.pyplot as plt
import seaborn as sns

with open("new_datadict", "rb") as input_file:
    data_dict = pickle.load(input_file, encoding='latin1') # Use if pickled in python 2

print("loaded data:\n", data_dict)

Ytrain = data_dict['Ytrain']
Xtrue = data_dict['Xtrue']
Yvalid = data_dict['Yvalid']

print("Shape of Ytrain:", Ytrain.shape)

# batch_size = 5
# print("Plotting loaded data...")
# plt.figure(figsize=(12, 12))
# plt.title("Training Time Series")
# plt.xlabel("Time")
# for i in range(Ytrain.shape[0]):
#     plt.subplot(Ytrain.shape[0] / batch_size, batch_size, i + 1)
#     plt.plot(Ytrain[i], c='red')
#     plt.plot(Xtrue[i], c='blue')
#     sns.despine()
#     plt.tight_layout()
# # plt.savefig(RLT_DIR + "Training Data")
# plt.show()

# print("Plotting loaded data...")
# plt.figure(figsize=(12, 12))
# plt.title("Training Time Series")
# plt.xlabel("Time")
# for i in range(Yvalid.shape[0]):
#     plt.subplot(Yvalid.shape[0] / batch_size, batch_size, i + 1)
#     plt.plot(Yvalid[i], c='red')
#     plt.plot(Xtrue[80+i], c='blue')
#     sns.despine()
#     plt.tight_layout()
# # plt.savefig(RLT_DIR + "Training Data")
# plt.show()

for i in range(Ytrain.shape[0]):
    print(Ytrain[i][0])
    print(Xtrue[i][0])