import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import torchvision as tv
import pickle
from time import time
#import src.model.resnet as resnet

from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from src.argument import parser, print_args

from transfer_utils import fine_tunify, transfer_datasets

from robustness import datasets, model_utils

from robustness import data_augmentation


class Trainer():
    def __init__(self, args, logger, attack, attack_IN):
        self.args = args
        self.logger = logger
        self.attack = attack
        self.attack_IN = attack_IN

    #def standard_train(self, model, tr_loader, va_loader=None):
    #    self.train(model, tr_loader, va_loader, False)

    #def adversarial_train(self, model, tr_loader, va_loader=None):
    #    self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.SGD(model.parameters(), args.learning_rate, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
        iter_per_epoch = math.ceil(50000.0/args.batch_size)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[iter_per_epoch*30, iter_per_epoch*50], 
                                                         gamma=0.1)
        _iter = 0
        test_acc_track = []
        adv_test_acc_track = []
        #IN_iter = iter(IN_tr_loader)
        begin_time = time()
        #criterion_kl = nn.KLDivLoss(size_average=False)
        #print(model.conv1.weight.grad)
        best_va_adv_acc = 0.0
        alpha = tensor2cuda(torch.Tensor([args.beta_dist_alpha]))
        beta = tensor2cuda(torch.Tensor([args.beta_dist_beta]))
        B_beta = torch.exp(torch.lgamma(alpha)+torch.lgamma(beta)-torch.lgamma(alpha+beta))


        max_entropy=tensor2cuda(torch.log(torch.tensor(100.)))
        for epoch in range(1, args.max_epoch+1):
            for data, label in tr_loader:
                #try:
                #    data_IN, label_IN = IN_iter.next()
                #except:
                #    IN_iter = iter(IN_tr_loader)
                #    data_IN, label_IN = IN_iter.next()
                #data_IN, label_IN = tensor2cuda(data_IN), tensor2cuda(label_IN)
                data, label = tensor2cuda(data), tensor2cuda(label)
                model.train()
                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    #data.requires_grad = True
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    #adv_data_IN = self.attack_IN.perturb_IN(data_IN, label_IN, 'mean', True)
                    #adv_data = data
                    output = model(adv_data)
                else:
                    output = model(data)

                #output_IN = model.forward_IN(adv_data_IN)

                rank_pred = torch.argsort(output, dim=-1, descending=True).detach()
                rank_gt = ((rank_pred-label.unsqueeze(1))==0).nonzero()[:,1]
                #print(rank_gt)
                rank_gt = rank_gt/100.0
                rank_gt_beta = 1.+torch.pow(rank_gt,alpha-1.)*torch.pow(1.-rank_gt,beta-1.)/B_beta
                
                #print(label_one_hot[0,1])
                #print(label_one_hot[0,2])

                loss_ind = F.cross_entropy(output, label, reduction='none')
                loss = torch.mean(loss_ind*rank_gt_beta)


                #loss_IN = F.cross_entropy(output_IN, label_IN)
                #loss_KD_robust = (1.0 / args.IN_batch_size) * criterion_kl(F.log_softmax(output, dim=1),F.softmax(t_model(adv_data), dim=1))
                opt.zero_grad()
                loss.backward()
                #print(model.conv1.weight.grad)
                opt.step()
                #pred_IN = torch.max(output_IN, dim=1)[1]
                #adv_acc_IN = evaluate(pred_IN.cpu().numpy(), label_IN.cpu().numpy()) * 100
                #print('KD loss: '+str(loss_KD_robust.cpu().detach().numpy()))
                if _iter % args.n_eval_step == 0:
                    t1 = time()

                    if adv_train:
                        with torch.no_grad():
                            model.eval()
                            stand_output = model(data)
                            model.train()
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:
                        
                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            model.eval()
                            adv_output = model(adv_data)
                            model.train()
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    t2 = time()

                    logger.info(f'epoch: {epoch}, iter: {_iter}, lr={opt.param_groups[0]["lr"]}, '
                                f'spent {time()-begin_time:.2f} s, tr_loss: {loss.item():.3f}')

                    logger.info(f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%')

                    # begin_time = time()

                    # if va_loader is not None:
                    #     va_acc, va_adv_acc = self.test(model, va_loader, True)
                    #     va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    #     logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                    #     logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    #         va_acc, va_adv_acc, time() - begin_time))
                    #     logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                # if _iter % args.n_store_image_step == 0:
                #     tv.utils.save_image(torch.cat([data.cpu(), adv_data.cpu()], dim=0), 
                #                         os.path.join(args.log_folder, f'images_{_iter}.jpg'), 
                #                         nrow=16)

                #if _iter % args.n_checkpoint_step == 0:
                #    file_name = os.path.join(args.model_folder, f'checkpoint_{_iter}.pth')
                #    save_model(model, file_name)

                _iter += 1
                # scheduler depends on training interation
                scheduler.step()

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n'+'='*20 +f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                    +'='*20)
                logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, spent: {t2-t1:.3f} s')
                logger.info('='*28+' end of evaluation '+'='*28+'\n')
                if va_adv_acc > best_va_adv_acc:
                    best_va_adv_acc = va_adv_acc
                    file_name = os.path.join(args.model_folder, f'checkpoint_best.pth')
                    save_model(model, file_name)
                #if epoch%10==0:
                #    file_name = os.path.join(args.model_folder, 'checkpoint_ep_'+str(epoch)+'.pth')
                #    save_model(model, file_name)
                test_acc_track.append(va_acc)
                adv_test_acc_track.append(va_adv_acc)
                pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_track.pkl','wb'))
                pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_track.pkl','wb'))
        file_name = os.path.join(args.model_folder, f'checkpoint_final.pth')
        save_model(model, file_name)

    def test(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        model.eval()
        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)
                
                output = model(data)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', 
                                                       False)
                    model.eval()
                    adv_output = model(adv_data)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num
        model.train()

        return total_acc / num , total_adv_acc / num

def main(args):

    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    #model = resnet.ResNet18()
    model_arch = 'resnet50'    
    model, _ = model_utils.make_and_restore_model(
                arch=model_arch,
                dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=False,
                add_custom_forward=True)
    
    while hasattr(model, 'model'):
        model = model.model
    model = fine_tunify.ft(
                model_arch, model, args.num_classes, 0)
    
    ds, (_,_) = transfer_datasets.make_loaders('cifar10', batch_size=10, workers=8, subset=50000)
    if type(ds) == int:
        print('new ds')
        new_ds = datasets.CIFAR(args.data_root)
        new_ds.num_classes = ds
        new_ds.mean = ch.tensor([0., 0., 0.])
        new_ds.std = ch.tensor([1.0, 1.0, 1.0])
        #new_ds.mean = ch.tensor([0.485, 0.456, 0.406])
        #new_ds.std = ch.tensor([0.229, 0.224, 0.225])
        ds = new_ds
    ds.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
    ds.std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, add_custom_forward=True)
    print(model) 

    #print(model)
    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)
    attack_IN = FastGradientSignUntargeted(model,
                                        args.epsilon/2.0,
                                        args.alpha/2.0,
                                        min_val=0,
                                        max_val=1,
                                        max_iters=args.k,
                                        _type=args.perturbation_type)
    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack, attack_IN)
    traindir = os.path.join(args.IN_data, 'train')
    valdir = os.path.join(args.IN_data, 'val')
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #train_dataset_IN = tv.datasets.ImageFolder(
    #    traindir, data_augmentation.TRAIN_TRANSFORMS_IMAGENET
    #        ) 
    #train_loader_IN = torch.utils.data.DataLoader(
    #    train_dataset_IN, batch_size=args.IN_batch_size, shuffle=True,
    #    num_workers=4, pin_memory=True, sampler=None)
    #swa_val_loader = torch.utils.data.DataLoader(
    #    val_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #    num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    #val_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(valdir, transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #    batch_size=args.batch_size, shuffle=False,
    #    num_workers=4, pin_memory=True)

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])
        tr_dataset = tv.datasets.CIFAR100(args.data_root, 
                                       train=True, 
                                       transform=transform_train, 
                                       download=False)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR100(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=False)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        te_dataset = tv.datasets.CIFAR100(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)

        print(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")

    else:
        raise NotImplementedError




if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
