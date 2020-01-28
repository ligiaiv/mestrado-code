import numpy
def sort_by_length(x, l, y):
        print(l)
        l_sorted, permutation = l.sort(0, descending=True)
        print("permutation",permutation)
        print(x)
        x_sorted = x[permutation]
        y_sorted = y[permutation]
        return x_sorted, l_sorted, y_sorted

def get_accuracy(hypos, refs):
        assert(len(hypos) == len(refs))
        correct = 0
        print("hypos", numpy.bincount(numpy.array(hypos)))
        for h, r in zip(hypos, refs):
                if h == r:
                        correct += 1
        total = len(refs)
        return correct * 100 / total

def evaluate_model(data_loader, model, set_name, sort=False):
        results_dev = []
        labels_dev = []
        for data in data_loader:
                if sort:  # needed for LSTM with pack_padded_sequences
                        x, l, y = sort_by_length(data['x'], data['length'], 
                                                 data['y'])
                else:
                        x, l, y = data['x'], data['length'], data['y']
                result = model(x, l)
                argmax = result.argmax(dim=1).cpu().numpy()
                results_dev.extend(list(argmax))
                labels_dev.extend(list(y.cpu().numpy()))
        accuracy = get_accuracy(results_dev, labels_dev)
        print("accuracy on " + set_name + ":", accuracy)