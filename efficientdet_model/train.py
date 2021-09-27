# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py

import argparse
import datetime
import os
import traceback
import psutil
import random

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer#, Augmenter#, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from bbaug.policies import policies


class ModelWithLoss(nn.Module):
    """Model with Focal Loss"""

    def __init__(self, model, debug=False):
        '''
        Initialize variables

        params
        :model (pytorch.model) -> model to perform a forward operation.
        :debug (bool) -> indicates if we want to debug, i.e., to save some images locally.
        '''
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        '''
        Performs a forward operation.

        params
        :imgs (torch.tensor) -> images in which we want to perform predictions.
        :annotations (torch.tensor) -> ground truth (bounding boxes) to calculate the loss.
        :obj_list -> list of objects to perform debugging.
        '''
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, 
                                                regression, 
                                                anchors, 
                                                annotations,
                                                imgs=imgs, 
                                                obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, 
                                                regression, 
                                                anchors, 
                                                annotations)
        return cls_loss, reg_loss



def train(opt, use_seed, aug_policy_container):
    '''
    Perform training of the model.

    params
    :opt (Class) -> parameters contained in the yml file.
    :use_seed (bool) -> indicates if we want to use a seed during training.
    '''

    # load project parameters
    params = Params(f'projects/{opt.project}.yml')
    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    #set a seed for everything
    if use_seed:
        #get seed or seeds (for the one or various experiments)
        seeds = [int(item) for item in opt.seed_values.split(' ')]
        my_seed = seeds[0]

        if torch.cuda.is_available():
            torch.cuda.manual_seed(my_seed)

        torch.manual_seed(my_seed)
        torch.cuda.manual_seed(my_seed)
        np.random.seed(my_seed)
        random.seed(my_seed)
        torch.backends.cudnn.deterministic = True

    # read paths to save weights
    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    # these are the standard sizes
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536] 
    #input_sizes = [1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280] 

    # define the training and validation sets
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), 
                                set=params.train_set,
                                transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Resizer(input_sizes[opt.compound_coef])]),
                                policy_container = aug_policy_container,
                                use_only_aug = opt.use_only_aug)

    training_generator = DataLoader(training_set, 
                                    batch_size= opt.batch_size,
                                    shuffle= True,
                                    drop_last= True,
                                    collate_fn= training_set.collater,
                                    num_workers= opt.num_workers)

    if opt.use_only_aug:
        val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), 
                                set=params.val_set,
                                transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                            Resizer(input_sizes[opt.compound_coef])]),
                                policy_container = aug_policy_container,
                                use_only_aug = opt.use_only_aug)
    else:
        val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), 
                                set=params.val_set,
                                transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                            Resizer(input_sizes[opt.compound_coef])]))

    val_generator = DataLoader(val_set, 
                                batch_size= opt.batch_size, 
                                shuffle= False,
                                drop_last= True,
                                collate_fn= val_set.collater,
                                num_workers= opt.num_workers)

    # define the model
    model = EfficientDetBackbone(num_classes=len(params.obj_list), 
                                compound_coef=opt.compound_coef,
                                ratios=eval(params.anchors_ratios), 
                                scales=eval(params.anchors_scales))

    # Load last weights from COCO
    #----------------------------------------------------
    if opt.load_weights is not None:
        weights_path = opt.load_weights

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        #print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        #Random initialization
        #print('[Info] initializing weights...')
        #init_weights(model)
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("error 1")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        return None

    # Freeze backbone if train head_only
    #----------------------------------------------------
    '''
    if opt.head_only:
        print("THIS SHOULD BE SET TO FALSE SINCE THIS IS SEMISUPERVISED")
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')
    '''

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    # set the optimizer and scheduler
    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5, min_lr=1e-6)

    # initial parameters
    epoch = 0
    best_loss = 1000.0
    best_epoch = 0
    last_step = 0
    step = max(0, last_step)
    model.train()
    num_iter_per_epoch = len(training_generator)

    # set the writer for the logs of tensorboard
    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # if something wrong happens, catch and save the last weights
    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    # move data to the gpu
                    if params.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    # perform the predictions
                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)

                    # calculate the loss
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    # calculate the gradients
                    loss.backward()

                    # update weights
                    optimizer.step()

                    #store the loss for later use in the scheduler
                    epoch_loss.append(float(loss))

                    # record in the logs
                    #----------------------------
                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1
                    #----------------------------

                    # save the model
                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_last.pth')
                        #save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_trained_weights.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            # use the scheduler -> maybe this will decrease the learning rate.
            scheduler.step(np.mean(epoch_loss))


            #---------- VALIDATION -------------
            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):

                # perform predictions and calculate the loss
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                # record in the logs
                #----------------------------
                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                #----------------------------


                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_best.pth')
                    #save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                    with open(os.path.join(opt.saved_path, f"best_epoch-d{opt.compound_coef}.txt"), "a") as my_file: 
                        my_file.write(f"Epoch:{epoch} / Step: {step} / Loss: {best_loss}\n") 

                # go back to training
                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break

    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_last.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    '''
    Save current weights of the model.

    params
    :model (torch.model) -> model to save.
    :name (string) -> filename to save this model.
    '''
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))



#               Section for handling parameters from user
#--------------------------------------------------------------------------------------------------------------------
class Params:
    """Read file with parameters"""
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    """Get all expected parameters"""
    parser = argparse.ArgumentParser('EfficientDet Pytorch')
    parser.add_argument('-p', '--project', type=str, default='coco')
    parser.add_argument('-c', '--compound_coef', type=int, default=0)
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--head_only', type=boolean_string, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw') # select optimizer for training, suggest using admaw until the very final stage then switch to sgd
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1) # Number of epoches between valing phases
    parser.add_argument('--save_interval', type=int, default=100) # Number of steps between saving
    parser.add_argument('--es_min_delta', type=float, default=0.0) # Early stopping's parameter: minimum change loss to qualify as an improvement
    parser.add_argument('--es_patience', type=int, default=0) # Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.
    parser.add_argument('--data_path', type=str, default='datasets/') # the root folder of dataset
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None) # whether to load weights from a checkpoint, set None to initialize, set last to load last checkpoint
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False) # whether visualize the predicted boxes of training, the output images will be in test
    #----------------------
    parser.add_argument('--use_seed', type=boolean_string, default=False)
    parser.add_argument('--seed_values', type=str, default="")
    parser.add_argument('--shuffle_ds', type=boolean_string, default=True)
    parser.add_argument('--policy', type=str, default="")
    parser.add_argument('--use_only_aug', type=boolean_string, default=False)
    parser.add_argument('--orig_height', type=float, default=0)
    parser.add_argument('--dest_height', type=float, default=0)

    args = parser.parse_args()
    return args


def throttle_cpu(cpu_list):
    """LIMIT THE NUMBER OF CPU TO PROCESS THE JOB. Useful when running in a server."""
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])



if __name__ == '__main__':
    #throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39]) 
    opt = get_args()

    # ask if we want to use a policy
    aug_policy_container = None
    random_state = None
    print(' ')
    print('#########################')
    print('POLICY NAME')
    print(opt.policy)
    print(opt.use_only_aug)
    print('#########################')
    if len(opt.policy) > 1:

        # select the policy
        if opt.policy == 'stac':
            aug_policy = policies.policies_STAC()
            aug_policy_container = policies.PolicyContainer(aug_policy, random_state = None if opt.use_seed == False else 42)

        if opt.policy == 'scaling':
            orig = opt.orig_height
            dest = opt.dest_height
            scaling_ratio = 1.0/(dest/orig)
            aug_policy = policies.policies_pineapple(scaling_ratio)
            aug_policy_container = policies.PolicyContainer(aug_policy, random_state = None if opt.use_seed == False else 42)
            print('#########################')
            print('Scaling magnitude')
            print(scaling_ratio)
            print('#########################')
        '''
        elif opt.policy == 'scaling_6m':
            ori = 5
            dest = 6
            scaling = 1/(dest/ori)
            aug_policy = policies.policies_pineapple(scaling)
            aug_policy_container = policies.PolicyContainer(aug_policy, random_state = None if opt.use_seed == False else 42)
            print('#########################')
            print('Scaling magnitude')
            print(scaling)
            print('#########################')
        elif opt.policy == 'scaling_7m':
            ori = 5
            dest = 7
            scaling = 1/(dest/ori)
            aug_policy = policies.policies_pineapple(scaling)
            aug_policy_container = policies.PolicyContainer(aug_policy, random_state = None if opt.use_seed == False else 42)
            print('#########################')
            print('Scaling magnitude')
            print(scaling)
            print('#########################')
        elif opt.policy == 'scaling_8m':
            ori = 5
            dest = 8
            scaling = 1/(dest/ori)
            aug_policy = policies.policies_pineapple(scaling)
            aug_policy_container = policies.PolicyContainer(aug_policy, random_state = None if opt.use_seed == False else 42)
            print('#########################')
            print('Scaling magnitude')
            print(scaling)
            print('#########################')
        elif opt.policy == 'scaling_9m':
            ori = 5
            dest = 9
            scaling = 1/(dest/ori)
            aug_policy = policies.policies_pineapple(scaling)
            aug_policy_container = policies.PolicyContainer(aug_policy, random_state = None if opt.use_seed == False else 42)
            print('#########################')
            print('Scaling magnitude')
            print(scaling)
            print('#########################')
        '''
    train(opt, opt.use_seed, aug_policy_container) 