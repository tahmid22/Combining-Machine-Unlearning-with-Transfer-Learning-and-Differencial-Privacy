import math

import numpy as np
import collections

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import utils
import time
import datetime
import mlxtend
from collections import defaultdict
from sklearn.naive_bayes import CategoricalNB
from mlxtend.classifier import EnsembleVoteClassifier

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
# import resnet as models
from torchvision.models import resnet18
import copy

from utils import SaveFile

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)
torch.cuda.empty_cache()

class SISA:
    def __init__(self, train_dataset, val_dataset, test_dataset, shards=1, slices=0, seed = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.N = len(train_dataset)
        self.shards = shards
        self.seed = seed
        self.estimators = []
        self.slices = slices

        self.shard_size = self.N//self.shards
        shard_size_arr = [self.shard_size]*self.shards
        self.shard_train_dataset_arr = torch.utils.data.random_split(self.train_dataset, shard_size_arr)
        self.affected_shards = []

        self.masks = {}

    def gen_random_seq(self, size): # sequence to delete random rows in our data
        delete_rows = np.random.choice(range(self.N), size=(size, 1), replace=False) # range start, stop + 1; # Max number of rows we can delete
        return delete_rows

    def fit(self, epsilon, fine_tune_percent, fine_tune_method, batch_size, epochs, workers, unlearn_requests=None):
        if self.shards != 0:
            # Initial fit without unlearning requests
            flag = not np.any(unlearn_requests)
            if flag:
                top1_val_accs_list = []
                top1_test_acc_list = []
                top5_val_accs_list = []
                top5_test_acc_list = []
                for shard_i, shard_train_dataset in enumerate(self.shard_train_dataset_arr):
                    if self.slices == 0:
                        print("Traing Shard_{}-------------------------------\n\
                        Shard Size: {}\n\
                        Epsilon: {}\n\
                        Fine-tune percent: {}\n\
                        Batch size: {}\n\
                        Epochs: {}\n".format(shard_i, len(shard_train_dataset), epsilon, fine_tune_percent, batch_size, epochs))

                        # Step-1: Load Dataset
                        train_dataloader = DataLoader(shard_train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
                        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=workers, pin_memory=False)
                        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=workers, pin_memory=False)
                        
                        # Step-2: Create the model and other releated variables
                        model, loss_fn, optimizer, scheduler, train_dataloader = self.create_model(train_dataloader, epsilon, fine_tune_percent, fine_tune_method, batch_size)
                        
                        # Step-3: Train the model epoch-times and keep track of best accuracy
                        best_acc = 0
                        best_model_wts = None
                        top1_val_accs = []
                        top5_val_accs = []
                        for epoch_i in range(epochs):
                            print(f"Epoch {epoch_i+1}\n-------------------------------")
                            self.train(train_dataloader, model, loss_fn, optimizer, scheduler, fine_tune_method)
                            epoch_acc_top1, epoch_acc_top5 = self.evaluate(val_dataloader, model, loss_fn, 'Validation')
                            top1_val_accs.append(epoch_acc_top1)
                            top5_val_accs.append(epoch_acc_top5)
                            if best_acc < epoch_acc_top1:
                                best_acc = epoch_acc_top1
                                best_model_wts = copy.deepcopy(model.state_dict())
                        print("Done traing shard_{}".format(shard_i))

                        model.load_state_dict(best_model_wts)
                        top1_test_acc, top5_test_acc = self.evaluate(test_dataloader, model, loss_fn, 'Test')

                        top1_val_accs_list.append(top1_val_accs)
                        top1_test_acc_list.append(top1_test_acc)
                        top5_val_accs_list.append(top5_val_accs)
                        top5_test_acc_list.append(top5_test_acc)
                        self.estimators.append(best_model_wts)

                    # Not experimenting with slicing
                    else:
                        print("Slicing not suppoted")
                        break

                save_file = SaveFile(self.shards, epsilon, fine_tune_percent, fine_tune_method, top1_val_accs_list, top1_test_acc_list, top5_val_accs_list, top5_test_acc_list, self.estimators)
                save_name = 'results_shards' + str(self.shards) + '_epsilon' + str(epsilon) + '_finetunepercent' + str(fine_tune_percent) + '_finetunemethod' + str(fine_tune_method)
                utils.save(save_file, save_name)
                #loaded = utils.load(save_name)
                #print(loaded.top1_val_accs_list)
                #print(loaded.top5_val_accs_list)

            # Case for retraining only the affected shards
            else:
                '''
                Skipping for now
                '''
                '''
                for x_shard, y_shard in zip(x_shards, y_shards):
                    bool_val = False
                    for item in x_shard:
                        for unlearn_request in unlearn_requests:
                            if unlearn_request[0] == item:
                                bool_val = True
                    if bool_val:
                        self.affected_shards.append(x_shard)
                affected_shards_x = np.array(self.affected_shards)

                for i, (x_shard,y_shard) in enumerate(zip(x_shards, y_shards)):
                    flag = False
                    for aff_x_shard in affected_shards_x:
                        if flag == True:
                            continue
                        elif np.array_equal(x_shard,aff_x_shard) != True:
                            for j in range(self.slices):
                                self.dict_train_time[i][j] = 0
                            continue
                        else:
                            flag = True
                            naive_b_wlearner = CategoricalNB(min_categories=256)
                            if self.slices == None:
                                start = datetime.datetime.now()
                                X_train_copy = np.copy(self.X_train)
                                y_train_copy = np.copy(self.y_train)
                                X_train_b_1 = X_train_copy[x_shard]
                                y_train_b_1 = y_train_copy[y_shard]
                                X_train_b = np.delete(X_train_b_1, x_shard, axis=0)
                                if X_train_b.size == 0:
                                    y_train_b = np.delete(y_train_b_1, y_shard)
                                    self.estimators.pop(i)
                                    continue
                                y_train_b = np.delete(y_train_b_1, y_shard, axis=0)
                                naive_b_wlearner.fit(X_train_b, y_train_b)
                                end = datetime.datetime.now()
                                diff = (end -start)
                                execution_time = diff.total_seconds() * 1000
                                self.dict_train_time[i] = execution_time
                                self.nb_learners[i] = naive_b_wlearner
                            else:
                                slices_in_x_shard = np.array_split(x_shard, self.slices) # slight innaccuracy because array_split will in some cases remove a tiny bit of data on the split; negligible
                                slices_in_y_shard = np.array_split(y_shard, self.slices)
                                # Currently in one slice at each shard 
                                for unlearn_request in unlearn_requests:
                                    for j, (x_ind_slice, y_ind_slice) in enumerate(zip(slices_in_x_shard, slices_in_y_shard)):
                                        start = datetime.datetime.now()
                                        if unlearn_request in x_ind_slice: # just omit the slice from the training of our model!!!!!
                                            self.dict_train_time[i][j] = 0
                                            continue          
                                        else: # feeding slices that are not in the unlearn request to our model
                                            X_train_slice = self.X_train[x_ind_slice]
                                            y_train_slice = self.y_train[y_ind_slice]
                                            naive_b_wlearner.partial_fit(X_train_slice, y_train_slice, classes=np.unique(y_train_slice))
                                        end = datetime.datetime.now()
                                        diff = (end -start)
                                        execution_time = diff.total_seconds() * 1000
                                        self.dict_train_time[i][j] = execution_time
                                self.estimators[i] = naive_b_wlearner
                '''

    def create_model(self, train_dataloader, epsilon, fine_tune_percent, finetune_method, epochs):
        # Create model
        model = resnet18(weights=None, num_classes=100)
        model = ModuleValidator.fix(model)  # BatchNorm -> GroupNorm
        model.load_state_dict(torch.load('pre-trained-model-gn-bs512-acc40'))  # TODO
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=10, bias=True)

        for name, param in model.named_parameters():
            param.requires_grad = False
            if 'conv' in name:
                if finetune_method >= 3:
                    param.requires_grad = True
                    total_param = torch.numel(param)
                    total_to_finetune = math.ceil(total_param * fine_tune_percent / 100)
                    values, indices = torch.topk(param.flatten(), total_to_finetune)

                    mask_unfrozen = torch.zeros_like(param.flatten()).cuda()
                    for i in indices:
                        mask_unfrozen[i] = 1
                    mask_unfrozen = torch.reshape(mask_unfrozen, param.shape)

                    mask_frozen = torch.ones_like(param.flatten()).cuda()
                    for i in indices:
                        mask_frozen[i] = 0
                    mask_frozen = torch.reshape(mask_frozen, param.shape)

                    self.masks[name] = (mask_unfrozen, mask_frozen)
            elif 'bn' in name:
                if finetune_method >= 2:
                    param.requires_grad = True
            elif 'fc' in name:
                if finetune_method >= 1:
                    param.requires_grad = True

        torch.nn.DataParallel(model).cuda()
        model = torch.nn.DataParallel(model).cuda()

        # Create loss function
        loss_fn = nn.CrossEntropyLoss().cuda()

        # Create optimizer
        fc_layer_names = ['fc.weight', 'fc.bias']
        non_fc_params = [x[1] for x in list(filter(lambda kv: kv[0] not in fc_layer_names, model.named_parameters()))]
        fc_params = [x[1] for x in list(filter(lambda kv: kv[0] in fc_layer_names, model.named_parameters()))]

        optimizer = torch.optim.SGD([
            {'params': non_fc_params},
            {'params': fc_params, 'lr': 0.8}
        ], lr=0.01, momentum=0.9)

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Make private
        privacy_engine = PrivacyEngine()
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_epsilon=epsilon,
            target_delta=0.00001,
            epochs=epochs,
            # noise_multiplier=15.0,
            max_grad_norm=1.0
        )

        return model, loss_fn, optimizer, scheduler, train_dataloader

    # def train(train_loader, model, criterion, optimizer, epoch):
    def train(self, dataloader: DataLoader, model, loss_fn, optimizer: torch.optim.Optimizer, scheduler, fine_tune_method):
        print("Learning rate:", scheduler.get_last_lr())
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            param_copies = {}
            if fine_tune_method >= 3:
                for name, param in model.named_parameters():
                    if 'conv' in name:
                        param_copies[name] = param.detach().clone()

            optimizer.step()

            if fine_tune_method >= 3:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'conv' in name:
                            #print('Changed param', param.flatten()[0])
                            mask_unfrozen, mask_frozen = self.masks[name[len('_module.module.'):]]
                            param_copy = param_copies[name]
                            #print('Unchanged param', param_copy.flatten()[0])
                            param.copy_(torch.add(torch.mul(param, mask_unfrozen), torch.mul(param_copy, mask_frozen)))
                            #print('New param', param.flatten()[0], '\n')

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        scheduler.step()

    def evaluate(self, dataloader: DataLoader, model, loss_fn, output_name):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
                pred = model(X)
                prec1, prec5 = accuracy(pred.data, y, topk=(1, 5))
                top1.update(prec1[0], X.size(0))
                top5.update(prec5[0], X.size(0))
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"{output_name} Error: \n Accuracy: {top1.avg.item() :>0.1f}% | {top5.avg.item() :>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return top1.avg.item(), top5.avg.item()

    def predict(self, X_test, X_train_copy, y_train_copy): # Sort of the aggregation method
        #y_test_hats = np.empty((len(self.nb_learners), len(X_test)))
        maj_vote = EnsembleVoteClassifier(clfs=self.estimators, voting='hard', fit_base_estimators=False) # Majority vote
        maj_vote.fit(X_train_copy, y_train_copy)
        for i in range(self.shards):
            sum = 0 
            for j in range(self.slices): # Sums up all the time it took to train each slice for our weak learner per shard
                sum+=self.dict_train_time[i][j]
            self.train_time_per_shard[i] = sum 
        for i in range(self.shards):
            print ('Execution time for training Shard %d : %d ms' %(i, self.train_time_per_shard[i]))
        self.reset_time()
        self.affected_shards = []
        return maj_vote.predict(X_test)
    
    def reset_time(self):
        for i in range(self.shards):
            self.train_time_per_shard[i] = 0

    def reset_learners(self):
        self.estimators = []
        self.fit()
        # Reset after intial fit to now calculate retrain time with starting point at 0 for all shards
        for i in range(self.shards):
            if self.slices != 0:
                for j in range(self.slices):
                    self.dict_train_time[i][j] = 0
            else:
                self.dict_train_time[i] = 0

    def validate(self, val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{ }]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

        return top1.avg


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            best_file = filename.replace('.tar', '_best.tar')
            shutil.copyfile(filename, best_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res