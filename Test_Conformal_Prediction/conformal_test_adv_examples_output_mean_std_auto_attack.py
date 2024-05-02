import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
from robustness import datasets, model_utils
from transfer_utils import fine_tunify, transfer_datasets


parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
parser.add_argument('--data', metavar='IN', help='path to Imagenet Val',default='/home/grads/ziquanliu2/data')
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=2000)
parser.add_argument('--num_classes', metavar='NCALIB', help='number of classes', default=100)
parser.add_argument('--save_score_name', help='path to save score', default='score')

#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/adv_pretrain_std_ft_R50_lr_0.01_wd_0.0001_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth')
# 
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/adv_pretrain_TRADES_R50_lr_0.001_wd_0.0001_beta_6.0_epoch_60_v1/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/at_conf/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/distil/adv_pretrain_R50_lr_0.01_wd_0.0001_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/distil_kl/adv_pretrain_R50_lr_0.01_wd_0.0001_lambda_dis_6.0_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/distil/adv_pretrain_R50_lr_0.01_wd_0.0001_lambda_dis_0.5_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/distil_only/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/std_pretrain_non_robust_ft_R50_lr_0.1_wd_0.0001_epoch_60_v2/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/distil_CP/adv_pretrain_R50_lr_0.003_wd_0.0001_epoch_10_attack_gt_label_v2_trial_1/cifar-10_linf/checkpoint_final.pth')
parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/distil_CP/adv_pretrain_R50_lr_0.003_wd_0.0001_epoch_60_attack_gt_label_v2_trial_1/cifar-10_linf/checkpoint_final.pth')
#parser.add_argument('--load_checkpoint', help='path to model', default='/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/distil_only/adv_pretrain_R50_lr_0.003_wd_0.0001_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_ep_10.pth')


parser.add_argument('--seed', metavar='SEED', type=int, help='random seed', default=0)


parser.add_argument('--epsilon', '-e', type=float, default=0.031, 
        help='maximum perturbation of adversaries (4/255=0.0157)')
parser.add_argument('--alpha', '-a', type=float, default=0.00784, 
        help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', '-k', type=int, default=100, 
        help='maximum iteration when generating adversarial examples')

# parser.add_argument('--epsilon', '-e', type=float, default=0.0, 
#         help='maximum perturbation of adversaries (4/255=0.0157)')
# parser.add_argument('--alpha', '-a', type=float, default=0.00784, 
#         help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
# parser.add_argument('--k', '-k', type=int, default=0, 
#         help='maximum iteration when generating adversarial examples')


parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
        help='the type of the perturbation (linf or l2)')
parser.add_argument('--use_adv_calib', help='whether use adv clib', default=False)

def output_mean_std(args, load_checkpoint_list, save_file):
    coverage_adv_all = []
    size_adv_all = []
    clean_acc_all = []
    clean_acc_top5_all = []
    robust_acc_all = []
    robust_acc_top5_all = []

    coverage_seed_list_normal = []
    size_seed_list_normal = []

    coverage_seed_list_adv = []
    size_seed_list_adv = []

    for load_checkpoint in load_checkpoint_list:
        # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92 
        transform = transforms.Compose([
                        transforms.ToTensor()
                    ])
        valdir = os.path.join(args.data, 'val')

        # Get the conformal calibration dataset
        dataset = torchvision.datasets.CIFAR100(args.data, train=False, transform=transform)
        print(len(dataset))
        all_test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


        
        cudnn.benchmark = True

        # Get the model 
        model_arch = 'resnet50'    
        model, _ = model_utils.make_and_restore_model(
                    arch=model_arch,
                    dataset=datasets.ImageNet(''), resume_path='', pytorch_pretrained=False,
                    add_custom_forward=True)
        
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify.ft(
                    model_arch, model, args.num_classes, 0)
        
        ds, (_,_) = transfer_datasets.make_loaders('cifar10', batch_size=10, workers=8, subset=50000)
        if type(ds) == int:
            print('new ds')
            new_ds = datasets.CIFAR(args.data)
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0., 0., 0.])
            new_ds.std = ch.tensor([1.0, 1.0, 1.0])
            #new_ds.mean = ch.tensor([0.485, 0.456, 0.406])
            #new_ds.std = ch.tensor([0.229, 0.224, 0.225])
            ds = new_ds
        ds.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        ds.std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        model_base, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, add_custom_forward=True)
    #
        checkpoint = torch.load(load_checkpoint)
        model_base.load_state_dict(checkpoint)

        model_base = torch.nn.DataParallel(model_base) 
        model_base.eval()

        # optimize for 'size' or 'adaptiveness'
        lamda_criterion = 'size'
        # allow sets of size zero
        allow_zero_sets = False 
        # use the randomized version of conformal
        randomized = True 

        adv_all_test_data = attack_image_AA(all_test_data_loader, model_base, args, print_bool=True)
        
        for set_seed in range(5):

            np.random.seed(seed=set_seed)
            torch.manual_seed(set_seed)
            torch.cuda.manual_seed(set_seed)
            random.seed(set_seed)
            adv_imagenet_calib_data, adv_imagenet_val_data = torch.utils.data.random_split(adv_all_test_data, [args.num_calib,len(adv_all_test_data)-args.num_calib])
            
            imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(dataset, [args.num_calib,len(dataset)-args.num_calib])

            # Initialize loaders

            adv_calib_loader = torch.utils.data.DataLoader(adv_imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            adv_val_loader = torch.utils.data.DataLoader(adv_imagenet_val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

            calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

            # Conformalize model
            model = ConformalModel(model_base, adv_calib_loader, args, alpha=0.1, lamda=0., kreg=0, randomized=randomized, allow_zero_sets=allow_zero_sets)

            print("Model calibrated and conformalized! Now evaluate over remaining data.")
            #_, _, coverage_normal, size_normal, size_std_normal, size_mean_normal = validate(val_loader, model, print_bool=True)
            
            #_, _, coverage_adv, size_adv, size_std_adv, size_mean_adv  = validate_adv_conf_score(val_loader, model, args, print_bool=True)
            clean_acc, clean_acc_top_5, coverage_normal, size_normal, coverage_list_normal, size_list_normal = validate(val_loader, model, print_bool=True)

            robust_acc, robust_acc_top_5, coverage_adv, size_adv, coverage_list_adv, size_list_adv  = validate(adv_val_loader, model, print_bool=True)
            print(load_checkpoint.split('/')[-3])
            print(load_checkpoint.split('/')[-1])
            print(args.epsilon*255.0)
            print('normal result:')
            print(coverage_normal)
            print(size_normal)

            print('adv result:')
            print(coverage_adv)
            print(size_adv)

            print("Complete!")
            
            coverage_adv_all.append(coverage_adv*100.0)
            size_adv_all.append(size_adv)
            clean_acc_all.append(clean_acc*100.)
            robust_acc_all.append(robust_acc*100.)
            clean_acc_top5_all.append(clean_acc_top_5*100.)
            robust_acc_top5_all.append(robust_acc_top_5*100.)

            coverage_seed_list_normal.append(coverage_list_normal)
            size_seed_list_normal.append(size_list_normal)

            coverage_seed_list_adv.append(coverage_list_adv)
            size_seed_list_adv.append(size_list_adv)
    
    # coverage_seed_list_normal = np.mean(np.asarray(coverage_seed_list_normal),axis=0)
    # size_seed_list_normal = np.mean(np.asarray(size_seed_list_normal),axis=0)
    # coverage_seed_list_adv = np.mean(np.asarray(coverage_seed_list_adv),axis=0)
    # size_seed_list_adv = np.mean(np.asarray(size_seed_list_adv),axis=0)


    coverage_seed_list_normal = np.asarray(coverage_seed_list_normal)
    size_seed_list_normal = np.asarray(size_seed_list_normal)
    coverage_seed_list_adv = np.asarray(coverage_seed_list_adv)
    size_seed_list_adv = np.asarray(size_seed_list_adv)

    print(size_seed_list_adv.shape)

    print(f' {np.mean(coverage_adv_all):.3f} ({np.std(coverage_adv_all):.3f}) & {np.mean(size_adv_all):.3f} ({np.std(size_adv_all):.3f}) & {np.mean(clean_acc_all):.3f} ({np.std(clean_acc_all):.3f}) & {np.mean(robust_acc_all):.3f} ({np.std(robust_acc_all):.3f}) & {np.mean(clean_acc_top5_all):.3f} ({np.std(clean_acc_top5_all):.3f}) & {np.mean(robust_acc_top5_all):.3f} ({np.std(robust_acc_top5_all):.3f})')
    result_string = f' {np.mean(coverage_adv_all):.3f} ({np.std(coverage_adv_all):.3f}) & {np.mean(size_adv_all):.3f} ({np.std(size_adv_all):.3f}) & {np.mean(clean_acc_all):.3f} ({np.std(clean_acc_all):.3f}) & {np.mean(robust_acc_all):.3f} ({np.std(robust_acc_all):.3f}) & {np.mean(clean_acc_top5_all):.3f} ({np.std(clean_acc_top5_all):.3f}) & {np.mean(robust_acc_top5_all):.3f} ({np.std(robust_acc_top5_all):.3f})'

    cp_curve = {'coverage_normal':coverage_seed_list_normal ,'size:normal':size_seed_list_normal, 'coverage_adv':coverage_seed_list_adv, 'size_adv':size_seed_list_adv, 'result':result_string}
    pickle.dump(cp_curve,open('result_AA_aps_no_adj/'+save_file+'.pkl','wb'))



if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness

    load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/CP_benchmark/transfer_robust_cifar100/MART/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    '/home/grads/ziquanliu2/ziquan_adv_robust/CP_benchmark/transfer_robust_cifar100/MART/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth'
    ]
    output_mean_std(args, load_checkpoint_list, 'MART')


    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'AT')


    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'AT_beta') # lr=0.0003


    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/entropy_minimization/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_ent_0.3_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/entropy_minimization/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_ent_0.3_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/entropy_minimization/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_ent_0.3_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'AT_EM') 


    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design_entropy_min/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_lambda_ent_0.3_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design_entropy_min/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_lambda_ent_0.3_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design_entropy_min/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_lambda_ent_0.3_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'AT_EM_beta')


    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cub/adv_pretrain_TRADES_R50_lr_0.01_wd_0.0001_beta_6.0_epoch_60_v1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cub/adv_pretrain_TRADES_R50_lr_0.01_wd_0.0001_beta_6.0_epoch_60_v1_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cub/adv_pretrain_TRADES_R50_lr_0.01_wd_0.0001_beta_6.0_epoch_60_v1_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]

    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/AT_fair_epoch_update_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_fair_1.0_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/AT_fair_epoch_update_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_fair_1.0_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/AT_fair_epoch_update_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_fair_1.0_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'Fair_AT_Entropy_Reg')


    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/TRADES_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/TRADES_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/TRADES_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'TRADES_Entropy_Reg')

    #load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/adv_pretrain_std_ft_R50_lr_0.01_wd_0.0001_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth']
    #output_mean_std(args, load_checkpoint_list, 'temp')

    #load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design_fair_entropy_min/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_lambda_fair_1.0_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth']
    #output_mean_std(args, load_checkpoint_list, 'Fair_AT_beta_Entropy_Reg')

    #load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/entropy_focal_loss_tcpr/adv_pretrain_R50_lr_0.01_wd_0.0001_lambda_focal_0.5_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth']
    #output_mean_std(args, load_checkpoint_list, 'AT_focal_TCPR')

    #load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/entropy_focal_loss_tcpr/adv_pretrain_R50_lr_0.01_wd_0.0001_lambda_focal_0.5_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth']
    #output_mean_std(args, load_checkpoint_list, 'AT_focal_TCPR')

    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/AT_fair_epoch_update_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_fair_1.0_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/AT_fair_epoch_update_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_fair_1.0_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/AT_fair_epoch_update_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_lambda_fair_1.0_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'Fair_AT_EM')

    # load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/TRADES_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/TRADES_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    # '/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/TRADES_entropy_min/adv_pretrain_R50_lr_0.001_wd_0.0001_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    # ]
    # output_mean_std(args, load_checkpoint_list, 'TRADES_EM')


    #load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/beta_trades/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    #'/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/beta_trades/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    #'/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/beta_trades/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    #]
    #output_mean_std(args, load_checkpoint_list, 'TRADES_beta')


    #load_checkpoint_list = ['/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design_fair_v2/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_lambda_fair_1.0_epoch_60_v2_trial_1/cifar-10_linf/checkpoint_final.pth',
    #'/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design_fair_v2/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_lambda_fair_1.0_epoch_60_v2_trial_2/cifar-10_linf/checkpoint_final.pth',
    #'/home/grads/ziquanliu2/ziquan_adv_robust/transfer_robust_cifar100/rank_focal_better_design_fair_v2/adv_pretrain_R50_lr_0.0003_wd_0.0001_alpha_1.1_beta_5.0_lambda_fair_1.0_epoch_60_v2_trial_3/cifar-10_linf/checkpoint_final.pth'
    #]
    #output_mean_std(args, load_checkpoint_list, 'Fair_AT_beta')
            
