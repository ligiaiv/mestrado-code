import numpy as np


def sort_by_length(x, l, y):
	# print("l",l)
	l_sorted, permutation = l.sort(0, descending=True)

	x_sorted = x[:, permutation]
	y_sorted = y[permutation]

	return x_sorted, l_sorted, y_sorted

# def get_accuracy(hypos, refs):

#         assert(len(hypos) == len(refs))
#         correct = 0
#         # print("hypos", numpy.bincount(numpy.array(hypos)))
#         for h, r in zip(hypos, refs):
#                 if h == r:
#                         correct += 1
#         total = len(refs)
#         return correct * 100 / total


def get_accuracy(hypos, refs):
	conf_matrix = np.matmul(hypos.transpose(), refs)
	assert(len(hypos) == len(refs))
	acc_weighted_sum = 0
	prec_weighted_sum = 0
	recall_weighted_sum = 0
	f1_weighted_sum = 0
	for categ in np.unique(np.concatenate((hypos, refs))):
		TP = conf_matrix[categ,categ]
		FP = conf_matrix[categ,:].sum() - TP
		FN = conf_matrix[:,categ].sum() - TP
		TN = conf_matrix.sum() - FP - FN - TP 
		# TN = ((hypos == refs) & (refs != categ))
		# FP = ((hypos != refs) & (hypos == categ))
		# FN = ((hypos != refs) & (hypos != categ))
		acc = (TP+TN)/(TP+TN+FP+FN)
		prec = TP/(TP+FP)
		recall = TP/(TP+FN)
		f1 = 2*recall*prec/(recall+prec) 
		n_categ = conf_matrix[:,categ].sum()
		acc_weighted_sum+=acc*n_categ
		f1_weighted_sum+=f1*n_categ
		recall_weighted_sum+=recall*n_categ
		prec_weighted_sum+=prec*n_categ
	total = conf_matrix.sum()
	
	acc_weighted_sum /=total
	prec_weighted_sum /=total
	recall_weighted_sum /=total
	f1_weighted_sum /=total
	# print("Hypos", hypos.shape)
	# print("Refs", refs.shape)
	# print("hypos", numpy.bincount(numpy.array(hypos)))
	return (acc_weighted_sum,prec_weighted_sum,recall_weighted_sum,f1_weighted_sum)


def evaluate_model_old(data_loader, model, set_name, sort=False):
	results_dev = []
	labels_dev = []
	# print("HERE: evaluate_model - before for loop")
	for data in data_loader:
		# print("HERE: evaluate_model - in for loop")
		x, l = data.text
		y = data.label
		if sort:  # needed for LSTM with pack_padded_sequences
			x, l, y = sort_by_length(x, l, y)
		# else:
		#         x, l, y = data['x'], data['length'], data['y']
		result = model(x, l)
		argmax = result.argmax(dim=1).cpu().numpy()
		results_dev.extend(list(argmax))
		labels_dev.extend(list(y.cpu().numpy()))
	print(type(results_dev))
	print(len(results_dev))
	accuracy = get_accuracy(results_dev, labels_dev)
	print("accuracy on " + set_name + ":", accuracy)
	return accuracy


def evaluate_model(data_loader, model, set_name, n_labels, sort=False):
	results_dev = np.ndarray((0, n_labels))
	labels_dev = np.ndarray((0, n_labels))
	for data in data_loader:
		# print("HERE: evaluate_model - in for loop")
		x, l = data.text
		y = data.label
		# n_labels = len(np.unique(data.label))
		print(n_labels)
		if sort:  # needed for LSTM with pack_padded_sequences
			x, l, y = sort_by_length(x, l, y)
		# else:
		#         x, l, y = data['x'], data['length'], data['y']
		result = model(x, l)
		argmax = result.argmax(dim=1).cpu().numpy()
		print(np.unique(y))
		ohk_results = np.eye(n_labels)[argmax]
		ohk_labels = np.eye(n_labels)[y]

		print("ohk", ohk_labels.shape)
		# print("A
		# RGMAX",argmax.shape)
		# print("RESULTS_DEV",results_dev.shape)
		results_dev = np.concatenate((results_dev, ohk_results), axis=0)
		# print("RESULTS_DEV",results_dev.shape)
		labels_dev = np.concatenate((labels_dev, ohk_labels))

		# labels_dev.extend(list(y.cpu().numpy()))
	print("TOTAL EACH REAL", labels_dev.sum(axis=0))
	accuracy = get_accuracy(results_dev, labels_dev)
	print("accuracy on " + set_name + ":", accuracy)
	return accuracy
