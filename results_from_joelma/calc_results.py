import pandas as pd
import numpy as np
import os, json,sys
from pprint import pprint
import matplotlib.pyplot as plt
from itertools import permutations 
import argparse


def read_file(FILE):
	print(FILE)
	with open(dir_path+"/"+FILE, "r") as read_file:
		in_js = json.load(read_file)
	return in_js

# in_df = pd.read_csv(dir_path+"/"+FILE,delimiter='|',dtype=float,header=None)
# in_np = in_df.values
# print(np.average(in_df.values))
def calc_metrics(in_js):
	metrics_values = []
	metrics = ["acc","prec","recall","f1"]
	for i,arr in enumerate(in_js["test"]):
		metrics_values.append(np.average(np.array(arr)))
		print(metrics[i] ,"is ",np.average(np.array(arr)))
	print('METRICS',metrics_values)
	return metrics_values
def plot_graph(in_js):
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

			train=square[0,:]
			val=square[1,:]
			print(val)
			plt.plot(x,train,color = colormap(i),linestyle='solid')
			plt.plot(x,val,color = colormap(i), linestyle = 'dashed')

			print(square.shape)
			i+=0.1
	else:
		train=train_val[0,:]
		val=train_val[1,:]

		plt.plot(x,train,color = "blue",linestyle='solid')
		plt.plot(x,val,color = "blue", linestyle = 'dashed')
		

	plt.show()
	print(conf_matrix)

def run_files(dir_path):
	df = pd.DataFrame(columns = ["filename","acc","prec","recall","f1"] )
	for filename in os.listdir(dir_path):
		if ".json" in filename:
			js = read_file(filename)
			results = calc_metrics(js) 
			results.insert(0,filename)
			new_row = pd.DataFrame([results],columns = ["filename","acc","prec","recall","f1"] )

			print("RESULTS",results)

			# df.loc[len(df)] = results
			df = df.append(new_row)
			print(df)
	
	df.to_csv(dir_path+"/results.csv")
	


parser = argparse.ArgumentParser()
parser.add_argument("--file", help="explicit file to read. If None, get first file in alphabetic order",action="store_true")
parser.add_argument("--final", help="if on, read files from resul",action="store_true")
parser.add_argument("--rf", help="if on, calc results of all files and save to csv",action="store_true")

args = parser.parse_args()




dir_path = os.path.dirname(os.path.realpath(__file__))
if args.final:
	dir_path = dir_path+"/resultados_finais"
if args.file:
	FILE = args.file
else:
	for (dirpath, dirnames, filenames) in os.walk(dir_path):
		print("filenames:",filenames)
	filenames = os.listdir(dir_path) 
	filenames.sort()
	FILE = filenames[-1]

print("\n")
print("\n\t"+FILE+"\n")
if args.rf:
	run_files(dir_path)
else:
	in_js = read_file(FILE)
	pprint(in_js["options"])

	plot_graph(in_js)

