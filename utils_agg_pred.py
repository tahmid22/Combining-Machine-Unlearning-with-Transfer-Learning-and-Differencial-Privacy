import pickle


class SaveFileAgg:
    def __init__(self, shards, epsilon, fine_tune_percent, fine_tune_method, top1_agg_acc_list, top5_agg_acc_list):
        self.shards = shards
        self.epsilon = epsilon
        self.fine_tune_percent = fine_tune_percent
        self.fine_tune_method = fine_tune_method
        self.top1_agg_acc_list = top1_agg_acc_list
        self.top5_agg_acc_list = top5_agg_acc_list


    def saveAgg(self):
        save_name = 'results_shards' + str(self.shards) + '_epsilon' + str(self.epsilon) + '_finetunepercent' + str(self.fine_tune_percent) + '_finetunemethod' + str(self.fine_tune_method) + "_aggprediction"
        file = open(save_name, 'wb')
        pickle.dump(self, file)
        file.close()

def load(name):
    file = open(name, 'rb')
    object = pickle.load(file)
    file.close()
    return object