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
	metrics = np.ndarray((4, 0))
	for categ in range(hypos.shape[1]):
		TP = conf_matrix[categ, categ]
		FP = conf_matrix[categ, :].sum() - TP
		FN = conf_matrix[:, categ].sum() - TP
		TN = conf_matrix.sum() - FP - FN - TP

		acc = (TP+TN)/(TP+TN+FP+FN)  # acc
		prec = TP/(TP+FP)  # prec
		recall = TP/(TP+FN)  # recall
		f1 = 2*recall*prec/(recall+prec)

		cat_metrics = np.expand_dims(np.array([acc, prec, recall, f1]), axis=1)
		metrics = np.hstack((metrics, cat_metrics))

	metrics = np.nan_to_num(metrics)

	[acc, prec, recall, f1] = (
		(metrics*conf_matrix.sum(axis=0)).sum(axis=1))/(conf_matrix.sum()).tolist()
	return (acc, prec, recall, f1)


def evaluate_model(data_loader, model, set_name, n_labels,architecture, sort=False):
	results_dev = np.ndarray((0, n_labels))
	labels_dev = np.ndarray((0, n_labels))
	for data in data_loader:

		x, l = data.text
		y = data.label
		# n_labels = len(np.unique(data.label))

		if sort:  # needed for LSTM with pack_padded_sequences
			x, l, y = sort_by_length(x, l, y)
		# else:
		#         x, l, y = data['x'], data['length'], data['y']
		if architecture == "bert":
			result = model(x.T)
		else:
			result = model(x, l)
		argmax = result.argmax(dim=1).cpu().numpy()
		
		ohk_results = np.eye(n_labels)[argmax]
		ohk_labels = np.eye(n_labels)[y]


		if ohk_labels.ndim == 1:
			ohk_labels = np.expand_dims(ohk_labels,axis = 0)
		
		# RGMAX",argmax.shape)

		results_dev = np.concatenate((results_dev, ohk_results), axis=0)

		labels_dev = np.concatenate((labels_dev, ohk_labels))

		# labels_dev.extend(list(y.cpu().numpy()))
	metrics = get_accuracy(results_dev, labels_dev)
	# print("metrics on " + set_name + " (acc, prec, recall, f1):", metrics)
	return metrics
