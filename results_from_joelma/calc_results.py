import pandas as pd
import numpy as np
import os, json,sys
from pprint import pprint
import matplotlib.pyplot as plt
from itertools import permutations 


# FILE = "/out_03-02-2020__19:12:31.json"
if len(sys.argv)>=2:
    FILE = sys.argv[1]
    print(FILE)
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

for (dirpath, dirnames, filenames) in os.walk(dir_path):
    print("filenames:",filenames)
filenames = os.listdir(dir_path) 
filenames.sort()
FILE = filenames[-1]

with open(dir_path+"/"+FILE, "r") as read_file:
    in_js = json.load(read_file)
print("\n")
pprint(in_js["options"])
# in_df = pd.read_csv(dir_path+"/"+FILE,delimiter='|',dtype=float,header=None)
# in_np = in_df.values
# print(np.average(in_df.values))
metrics = ["acc","prec","recall","f1"]
for i,arr in enumerate(in_js["test"]):
    print(metrics[i] ,"is ",np.average(np.array(arr)))
train_val = np.array(in_js["train_val"])
conf_matrix = np.array(in_js.get("confmat","No conf matrix"))
print(train_val.shape)
#
#   Plot for K = 0
#
k0_train = train_val[0,:]
k0_val = train_val[1,:]
# k0_train = train_val[0,:,1]
# k0_val = train_val[1,:,1]
print(k0_train.shape)
# quit()
x = np.arange(1,k0_train.shape[0]+1)
colormap = plt.cm.get_cmap("Spectral")
i = 0
if len(train_val.shape)>2:
    for square in np.transpose(train_val,(2,0,1)):
        # color = plt.cm.get_cmap("Spectral")(i)

        train=square[0,:]
        val=square[1,:]
        print(val)
        plt.plot(x,train,color = colormap(i),linestyle='solid')
        plt.plot(x,val,color = colormap(i), linestyle = 'dashed')
        
        # plt.show()
        # quit()
        print(square.shape)
        i+=0.1
else:
    train=train_val[0,:]
    val=train_val[1,:]
    # print(val)
    plt.plot(x,train,color = "blue",linestyle='solid')
    plt.plot(x,val,color = "blue", linestyle = 'dashed')
    
    # plt.show()
    # quit()
    # print(square.shape)
plt.show()
print(conf_matrix)
quit()
# x = [2, 4, 6]
# y = [1, 3, 5]
# plt.plot(x, y)
# plt.show()

# pprint(in_js)