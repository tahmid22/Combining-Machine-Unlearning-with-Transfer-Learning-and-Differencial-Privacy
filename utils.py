import pickle


class SaveFile:
    def __init__(self, shards, epsilon, fine_tune_percent, fine_tune_method, top1_val_accs_list, top1_test_acc_list, estimator_wts_list):
        self.shards = shards
        self.epsilon = epsilon
        self.fine_tune_percent = fine_tune_percent
        self.fine_tune_method = fine_tune_method
        self.top1_val_accs_list = top1_val_accs_list
        self.top1_test_acc_list = top1_test_acc_list
        self.estimator_wts_list = estimator_wts_list


def save(object, name):
    file = open(name, 'wb')
    pickle.dump(object, file)
    file.close()


def load(name):
    file = open(name, 'rb')
    object = pickle.load(file)
    file.close()
    return object
