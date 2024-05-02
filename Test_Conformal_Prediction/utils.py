import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import pdb
from src.attack import FastGradientSignUntargeted
from scipy.special import softmax
import scipy.io as sio
from autoattack import AutoAttack

def sort_sum(scores):
    I = scores.argsort(axis=1)[:,::-1] # reverse the order, large to small
    ordered = np.sort(scores,axis=1)[:,::-1] # reverse the order
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def attack_image(val_loader, model, args, print_bool):
    # attack = FastGradientSignUntargeted(model, 
    #                                     1.0, 
    #                                     0.2, 
    #                                     min_val=0, 
    #                                     max_val=1, 
    #                                     max_iters=args.k, 
    #                                     _type='l2')

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon*1., 
                                        args.alpha*1., 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    adv_img = torch.zeros((len(val_loader.dataset), 3, 32, 32)) # 1000 classes in Imagenet.
    adv_labels = torch.zeros((len(val_loader.dataset),)) # 1000 classes in Imagenet.

    with torch.no_grad():
        
        # switch to evaluate mode
        model.eval()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            adv_data = attack.perturb_no_conf(x.cuda(), target, 'mean',False)

            model.eval()

            adv_img[N:N+len(x),:,:,:] = adv_data.detach().cpu()
            adv_labels[N:N+len(x)] = target.cpu()
            N += len(x)
        
    adv_dataset = torch.utils.data.TensorDataset(adv_img, adv_labels.long()) 
    #pickle.dump([adv_img,adv_img],open('test_adv_examples_std_model.pkl','wb'))
    return adv_dataset

def attack_image_clean(val_loader, model, args, print_bool):
    # attack = FastGradientSignUntargeted(model, 
    #                                     1.0, 
    #                                     0.2, 
    #                                     min_val=0, 
    #                                     max_val=1, 
    #                                     max_iters=args.k, 
    #                                     _type='l2')


    adv_img = torch.zeros((len(val_loader.dataset), 3, 32, 32)) # 1000 classes in Imagenet.
    adv_labels = torch.zeros((len(val_loader.dataset),)) # 1000 classes in Imagenet.

    with torch.no_grad():
        
        # switch to evaluate mode
        model.eval()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            x = x.cuda()
            # compute output

            adv_img[N:N+len(x),:,:,:] = x.detach().cpu()
            adv_labels[N:N+len(x)] = target.cpu()
            N += len(x)
        
    adv_dataset = torch.utils.data.TensorDataset(adv_img, adv_labels.long()) 
    #pickle.dump([adv_img,adv_img],open('test_adv_examples_std_model.pkl','wb'))
    return adv_dataset


def attack_image_AA(val_loader, model, args, print_bool):
    # attack = FastGradientSignUntargeted(model, 
    #                                     1.0, 
    #                                     0.2, 
    #                                     min_val=0, 
    #                                     max_val=1, 
    #                                     max_iters=args.k, 
    #                                     _type='l2')

    adversary = AutoAttack(model, norm='Linf', eps=0.031, version='standard', verbose=False, log_path = './temp')
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']

    adv_img = torch.zeros((len(val_loader.dataset), 3, 32, 32)) # 1000 classes in Imagenet.
    adv_labels = torch.zeros((len(val_loader.dataset),)) # 1000 classes in Imagenet.

    with torch.no_grad():
        
        # switch to evaluate mode
        model.eval()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            x = x.cuda()
            # compute output
            adv_data = adversary.run_standard_evaluation(x, target, bs=len(x))
            model.eval()

            adv_img[N:N+len(x),:,:,:] = adv_data.detach().cpu()
            adv_labels[N:N+len(x)] = target.cpu()
            N += len(x)
        
    adv_dataset = torch.utils.data.TensorDataset(adv_img, adv_labels.long()) 
    #pickle.dump([adv_img,adv_img],open('test_adv_examples_std_model.pkl','wb'))
    return adv_dataset


def validate_adv_conf_score(val_loader, model, args, print_bool):
    # attack = FastGradientSignUntargeted(model, 
    #                                     1.0, 
    #                                     0.2, 
    #                                     min_val=0, 
    #                                     max_val=1, 
    #                                     max_iters=args.k, 
    #                                     _type='l2')

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon*1., 
                                        args.alpha*1., 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        size_list = []
        end = time.time()
        N = 0
        E = np.array([])
        E_no_rand = np.array([])
        gt_rank = np.array([])
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            adv_data = attack.perturb(x.cuda(), target, 'mean',False)
            model.eval()

            output, S = model(adv_data.cuda())
            rank_pred = torch.argsort(output, dim=-1, descending=True)
            rank_gt = ((rank_pred-target.unsqueeze(1))==0).nonzero()[:,1]

            logits_numpy = output.detach().cpu().numpy()
            scores = softmax(logits_numpy/model.T.item(), axis=1)
            I, ordered, cumsum = sort_sum(scores)
            #print(model.penalties)
            E = np.concatenate((E,giq(scores,target,I=I,ordered=ordered,cumsum=cumsum,penalties=model.penalties,randomized=True, allow_zero_sets=True)))
            
            E_no_rand = np.concatenate((E_no_rand,giq(scores,target,I=I,ordered=ordered,cumsum=cumsum,penalties=model.penalties,randomized=False, allow_zero_sets=True)))
            
            gt_rank = np.concatenate((gt_rank,rank_gt.cpu().numpy()))


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S[0], target)
            for j in range(len(x)):
                size_list.append(float(S[0][j].shape[0]))

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('') #Endline
    #print(E)
    #print(E.size)
    train_result = {'cscore':E,'cscore_no_rand':E_no_rand,'gt_rank':gt_rank}
    pickle.dump(train_result,open('visualize/'+args.load_checkpoint.split('/')[-3]+'.pkl','wb'))
    #sio.savemat('visualize/'+args.load_checkpoint.split('/')[-3]+'.mat',{'cscore':E,'cscore_no_rand':E_no_rand,'gt_rank':gt_rank})

    return top1.avg, top5.avg, coverage.avg, size.avg, np.std(size_list), np.mean(size_list)

def class_wise_acc(val_loader, model, args, print_bool):
    # attack = FastGradientSignUntargeted(model, 
    #                                     1.0, 
    #                                     0.2, 
    #                                     min_val=0, 
    #                                     max_val=1, 
    #                                     max_iters=args.k, 
    #                                     _type='l2')

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon*1., 
                                        args.alpha*1., 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage_list = []
        size_cp_list = []
        size_list=[0.]
        num_threshold = 80
        for i in range(num_threshold):
            coverage_list.append(AverageMeter('RAPS coverage'+str(i)))
            size_cp_list.append(AverageMeter('RAPS size'+str(i)))
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        target_all = []
        acc_all = []
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            adv_data = attack.perturb(x.cuda(), target, 'mean',False)
            model.eval()

            output, S = model(adv_data.cuda())
            # measure accuracy and record loss
            prec1 = accuracy_vec(output, target, topk=(1,))
            target_all = target_all + target.cpu().numpy().tolist()
            #print(prec1)
            #print(prec1[0].cpu().numpy().tolist())
            acc_all = acc_all + prec1[0].cpu().numpy().tolist()[0]

            #print(len(target_all))
            #print(len(acc_all))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
    target_all_np = np.asarray(target_all)
    acc_all_np = np.asarray(acc_all)
    class_acc = []
    for i in range(args.num_classes):
        class_acc.append(np.sum(acc_all_np[target_all_np==i])/np.sum(target_all_np==i))
            
    
    
    return class_acc




def validate_adv(val_loader, model, args, print_bool):
    # attack = FastGradientSignUntargeted(model, 
    #                                     1.0, 
    #                                     0.2, 
    #                                     min_val=0, 
    #                                     max_val=1, 
    #                                     max_iters=args.k, 
    #                                     _type='l2')

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon*1., 
                                        args.alpha*1., 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage_list = []
        size_cp_list = []
        size_list=[0.]
        num_threshold = 100
        for i in range(num_threshold):
            coverage_list.append(AverageMeter('RAPS coverage'+str(i)))
            size_cp_list.append(AverageMeter('RAPS size'+str(i)))
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            adv_data = attack.perturb(x.cuda(), target, 'mean',False)
            model.eval()

            output, S = model(adv_data.cuda())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            for ind, S_ind in enumerate(S):
                cvg, sz = coverage_size(S_ind, target)
                #for j in range(len(x)):
                #    size_list.append(float(S[j].shape[0]))
                coverage_list[ind].update(cvg, n=x.shape[0])
                size_cp_list[ind].update(sz, n=x.shape[0])
            

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage_list[15].val:.3f} ({coverage_list[15].avg:.3f}) | Size@RAPS: {size_cp_list[15].val:.3f} ({size_cp_list[15].avg:.3f})', end='')
    if print_bool:
        print('') #Endline
    coverage_avg = []
    #print('coverage of different threshold:')
    for i in range(num_threshold):
        coverage_avg_ind = coverage_list[i].avg
        #print(str(coverage_avg_ind)+', '+str(size_cp_list[i].avg))
        coverage_avg.append(abs(coverage_avg_ind-0.9))
    chosen_threshold = coverage_avg.index(min(coverage_avg))
    

    return top1.avg, top5.avg, coverage_list[chosen_threshold].avg, size_cp_list[chosen_threshold].avg, np.std(size_list), np.mean(size_list)


def conf_pred_label(val_loader, model, print_bool):
    CP_label=[]
    CP_prob=[]
    

    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        size_list = []
        end = time.time()
        N = 0
        for i, (x, target, index) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            logits_numpy = output.detach().cpu().numpy()
            scores = softmax(logits_numpy/model.T.item(), axis=1)

            I, ordered, cumsum = sort_sum(scores)


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)
            for j in range(len(x)):
                size_list.append(float(S[j].shape[0]))
                CP_prob.append(ordered[j,:S[j].shape[0]])
            print(index)

            CP_label = CP_label + S

            print(len(CP_label))
            print(len(CP_prob))


            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('') #Endline
    CP_label_one_hot = np.zeros((len(CP_label),100))
    CP_prob_one_hot = np.zeros((len(CP_label),100))

    for i in range(len(CP_label)):
        print(i)
        CP_label_one_hot[i,CP_label[i]]=1
        CP_prob_one_hot[i,CP_label[i]] = CP_prob[i]
    #print(CP_label_one_hot[0:10])
    #print(CP_prob_one_hot[0:10])

    #pickle.dump(CP_prob_one_hot,open('CP_prob_one_hot_pt_epoch_20.pkl','wb'))


    return top1.avg, top5.avg, coverage.avg, size.avg, np.std(size_list), np.mean(size_list)


def validate_adv_logits(val_loader, model, args, print_bool):
    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon*1., 
                                        args.alpha*1., 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        size_list = []
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            adv_data = attack.perturb(x.cuda(), target, 'mean',False)
            model.eval()
            output, S = model(x.cuda())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)
            for j in range(len(x)):
                size_list.append(float(S[j].shape[0]))

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('') #Endline

    return top1.avg, top5.avg, coverage.avg, size.avg


def validate_logits(val_loader, model, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        size_list = []
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)
            for j in range(len(x)):
                size_list.append(float(S[j].shape[0]))

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('') #Endline

    return top1.avg, top5.avg, coverage.avg, size.avg


def validate(val_loader, model, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage_list = []
        size_cp_list = []
        num_threshold = 300
        for i in range(num_threshold):
            coverage_list.append(AverageMeter('RAPS coverage'+str(i)))
            size_cp_list.append(AverageMeter('RAPS size'+str(i)))

        # coverage = AverageMeter('RAPS coverage 1')
        # coverage = AverageMeter('RAPS coverage 2')
        # coverage = AverageMeter('RAPS coverage 3')
        # coverage = AverageMeter('RAPS coverage 4')
        # coverage = AverageMeter('RAPS coverage 5')
        # coverage = AverageMeter('RAPS coverage 6')


        # switch to evaluate mode
        model.eval()
        size_list = [0.0]
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            for ind, S_ind in enumerate(S):
                cvg, sz = coverage_size(S_ind, target)
                #for j in range(len(x)):
                #    size_list.append(float(S[j].shape[0]))
                coverage_list[ind].update(cvg, n=x.shape[0])
                size_cp_list[ind].update(sz, n=x.shape[0])

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage_list[0].val:.3f} ({coverage_list[0].avg:.3f}) | Size@RAPS: {size_cp_list[0].val:.3f} ({size_cp_list[0].avg:.3f})', end='')
    if print_bool:
        print('') #Endline
    coverage_avg_diff = []
    coverage_avg = []
    size_avg = []
    #print('coverage of different threshold:')
    for i in range(num_threshold):
        coverage_avg_ind = coverage_list[i].avg
        coverage_avg.append(coverage_avg_ind)
        size_avg.append(size_cp_list[i].avg)
        
        #print(str(coverage_avg_ind)+', '+str(size_cp_list[i].avg))
        coverage_avg_diff.append(abs(coverage_avg_ind-0.9))
    chosen_threshold = coverage_avg_diff.index(min(coverage_avg_diff))
    
    return top1.avg, top5.avg, coverage_list[chosen_threshold].avg, size_cp_list[chosen_threshold].avg, coverage_avg, size_avg

def coverage_size(S,targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]

def accuracy_vec(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float()
        res.append(correct_k)
    return res

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).cuda()
    targets = torch.cat([torch.Tensor([int(x[1])]) for x in data], dim=0).long()
    return imgs, targets

def split2ImageFolder(path, transform, n1, n2):
    dataset = torchvision.datasets.ImageFolder(path, transform)
    data1, data2 = torch.utils.data.random_split(dataset, [n1, len(dataset)-n1])
    data2, _ = torch.utils.data.random_split(data2, [n2, len(dataset)-n1-n2])
    return data1, data2

def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    return data1, data2

def get_model(modelname):
    if modelname == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True, progress=True)

    elif modelname == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True, progress=True)

    elif modelname == 'ResNet101':
        model = torchvision.models.resnet101(pretrained=True, progress=True)

    elif modelname == 'ResNet152':
        model = torchvision.models.resnet152(pretrained=True, progress=True)

    elif modelname == 'ResNeXt101':
        model = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)

    elif modelname == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True, progress=True)

    elif modelname == 'ShuffleNet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True, progress=True)

    elif modelname == 'Inception':
        model = torchvision.models.inception_v3(pretrained=True, progress=True)

    elif modelname == 'DenseNet161':
        model = torchvision.models.densenet161(pretrained=True, progress=True)

    else:
        raise NotImplementedError

    model.eval()
    model = torch.nn.DataParallel(model).cuda()

    return model

# Computes logits and targets from a model and loader
def get_logits_targets(model, loader, args):
    logits = torch.zeros((len(loader.dataset), 100)) # 1000 classes in Imagenet.
    labels = torch.zeros((len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):

            batch_logits = model(x.cuda()).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits


def get_logits_targets_adv(model, loader, args):
    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    attack_v2 = FastGradientSignUntargeted(model, 
                                        args.epsilon/2.0, 
                                        args.alpha/2.0, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    logits = torch.zeros((1*len(loader.dataset), 100)) # 1000 classes in Imagenet.
    labels = torch.zeros((1*len(loader.dataset),))
    i = 0
    print(f'Computing logits for model (only happens once).')
    with torch.no_grad():
        for x, targets in tqdm(loader):
            adv_data = attack.perturb_normal_output(x.cuda(), targets.cuda(), 'mean',False)
            model.eval()
            batch_logits = model(adv_data).detach().cpu()
            logits[i:(i+x.shape[0]), :] = batch_logits
            labels[i:(i+x.shape[0])] = targets.cpu()
            i = i + x.shape[0]
        # for x, targets in tqdm(loader):
        #     adv_data = attack_v2.perturb_normal_output(x.cuda(), targets.cuda(), 'mean',False)
        #     batch_logits = model(adv_data).detach().cpu()
        #     logits[i:(i+x.shape[0]), :] = batch_logits
        #     labels[i:(i+x.shape[0])] = targets.cpu()
        #     i = i + x.shape[0]
    
    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits

def get_logits_dataset(modelname, datasetname, datasetpath, cache=str(pathlib.Path(__file__).parent.absolute()) + '/experiments/.cache/'):
    fname = cache + datasetname + '/' + modelname + '.pkl' 

    # If the file exists, load and return it.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    # Else we will load our model, run it on the dataset, and save/return the output.
    model = get_model(modelname)



    transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])

    # transform = transforms.Compose([
    #                 transforms.Resize(256),
    #                 transforms.CenterCrop(224),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std =[0.229, 0.224, 0.225])
    #                 ])
    
    dataset = torchvision.datasets.ImageFolder(datasetpath, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=False, pin_memory=True)

    # Get the logits and targets
    dataset_logits = get_logits_targets(model, loader)

    # Save the dataset 
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as handle:
        pickle.dump(dataset_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_logits


def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1]) # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1] * \
                    (tau-(cumsum[i,sizes_base[i]-1]-ordered[i,sizes_base[i]-1])-penalties_cumsum[0,sizes_base[i]-1]) # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[sizes == 0] = 1 # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S

# Get the 'p-value'
def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    print(idx)
    print(cumsum)

    if not randomized:
        return tau_nonrandom + penalty[0]
    
    U = np.random.random()

    if idx == (0,0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0] 
    else:
        return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()

# Gets the histogram of Taus. 
def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],penalties[0,:],randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E

### AUTOMATIC PARAMETER TUNING FUNCTIONS
def pick_kreg(paramtune_logits, alpha):
    gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits])
    kstar = np.quantile(gt_locs_kstar, 1-alpha, interpolation='higher') + 1
    return kstar 

def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
    # Calculate lamda_star
    best_size = iter(paramtune_loader).__next__()[0][1].shape[0] # number of classes 
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]: # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
        if sz_avg < best_size:
            best_size = sz_avg
            lamda_star = temp_lam
    return lamda_star

def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets, strata=[[0,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]):
    # Calculate lamda_star
    lamda_star = 0
    best_violation = 1
    # Use the paramtune data to pick lamda.  Does not violate exchangeability.
    for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3, 2e-3]: # predefined grid, change if more precision desired.
        conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam, randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
        curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
        if curr_violation < best_violation:
            best_violation = curr_violation 
            lamda_star = temp_lam
    return lamda_star

def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size, lamda_criterion):
    num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
    paramtune_logits, calib_logits = tdata.random_split(calib_logits, [num_paramtune, len(calib_logits)-num_paramtune])
    calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
    paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

    if kreg == None:
        kreg = pick_kreg(paramtune_logits, alpha)
    if lamda == None:
        if lamda_criterion == "size":
            lamda = pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
        elif lamda_criterion == "adaptiveness":
            lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
    return kreg, lamda, calib_logits

def get_violation(cmodel, loader_paramtune, strata, alpha):
    df = pd.DataFrame(columns=['size', 'correct'])
    for logit, target in loader_paramtune:
        # compute output
        output, S = cmodel(logit) # This is a 'dummy model' which takes logits, for efficiency.
        # measure accuracy and record loss
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy()) 
        correct = np.zeros_like(size)
        for j in range(correct.shape[0]):
            correct[j] = int( target[j] in list(S[j]) )
        batch_df = pd.DataFrame({'size': size, 'correct': correct})
        df = df.append(batch_df, ignore_index=True)
    wc_violation = 0
    for stratum in strata:
        temp_df = df[ (df['size'] >= stratum[0]) & (df['size'] <= stratum[1]) ]
        if len(temp_df) == 0:
            continue
        stratum_violation = abs(temp_df.correct.mean()-(1-alpha))
        wc_violation = max(wc_violation, stratum_violation)
    return wc_violation # the violation



