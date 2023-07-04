import torch.backends.cudnn as cudnn
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from datetime import datetime
from data_loader import DataLoader_train_pre, DataLoader_test_pre
import random
from networks.unet_ConvNeXt import convnext_tiny


def checkpoint(net, epoch, name, type):
    save_model_path = os.path.join(args.save_model_path, args.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = '{}-{:03d}-{}.pth'.format(name, epoch, type)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(),
               save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str,
                    default='./datasets/DECT_data_denoised/train', help='root dir for train data')
parser.add_argument('--test_data_dir', type=str,
                    default='./datasets/DECT_data_denoised/test', help='root dir for test data')
parser.add_argument('--label_dir', type=str,
                    default='./datasets/DECT_data_denoised/label', help='root dir for label data')
parser.add_argument('--log_name', type=str,
                    default='material_decomposition', help='network function')
parser.add_argument('--save_model_path', type=str,
                    default='./pre_results_decomposition', help='dir for network weight')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training')
parser.add_argument("--DATE_FORMAT", type=str, default='%A_%d_%B_%Y_%Hh_%Mm_%Ss', help='time format')

parser.add_argument("--Ablation_mix", type=bool, default=False, help='Ablation study for incorporation')
parser.add_argument("--Ablation_att", type=bool, default=True, help='Ablation study for attention')

args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

base_lr = args.base_lr
num_classes = args.num_classes
batch_size = args.batch_size * args.n_gpu

systime = datetime.now().strftime(args.DATE_FORMAT)

material_dict = {0: 'Adipose', 1: 'Muscle', 2: 'Iodine', 3: 'Air', 4: '100kv', 5: '140kv'}

TrainingDataset = DataLoader_train_pre(args.train_data_dir, args.label_dir)
TestDataset = DataLoader_test_pre(args.test_data_dir)

TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

TestLoader = DataLoader(dataset=TestDataset,
                        num_workers=8,
                        batch_size=2,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=True)

model = convnext_tiny(num_classes=4, Ablation_mix=args.Ablation_mix, Ablation_att=args.Ablation_att).cuda()

if args.n_gpu > 1:
    model = nn.DataParallel(model)
model.train()
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

if args.resume:
    recent_folder = most_recent_folder(os.path.join(args.save_model_path, args.log_name), fmt=args.DATE_FORMAT)
    if not recent_folder:
        raise Exception('no recent folder were found')

    checkpoint_path = os.path.join(args.save_model_path, args.log_name, recent_folder)
    best_weights = best_acc_weights(os.path.join(args.save_model_path, args.log_name, recent_folder))
    if best_weights:
        weights_path = os.path.join(args.save_model_path, args.log_name, recent_folder, best_weights)
        print('found best acc weights file:{}'.format(weights_path))
        print('load best training file...')
        model.load_state_dict(torch.load(weights_path))

    recent_weights_file = most_recent_weights(os.path.join(args.save_model_path, args.log_name, recent_folder))
    if not recent_weights_file:
        raise Exception('no recent weights file were found')
    weights_path = os.path.join(args.save_model_path, args.log_name, recent_folder, recent_weights_file)
    print('loading weights file {} to resume training.....'.format(weights_path))
    model.load_state_dict(torch.load(weights_path))

    resume_epoch = last_epoch(os.path.join(args.save_model_path, args.log_name, recent_folder))
else:
    checkpoint(model, 0, "model", "regular")

print('init finish')

iter_num = 0
max_epoch = args.max_epochs
max_iterations = args.max_epochs * len(TrainingLoader)
best_performance = 0.0
Loss1 = nn.MSELoss()
iterator = tqdm(range(max_epoch), ncols=70)
for epoch_num in iterator:
    if args.resume:
        if epoch_num <= resume_epoch:
            continue

    loss_train_sum = 0

    for i_batch, ima in enumerate(TrainingLoader):
        st = time.time()
        image_batch, label_batch = ima
        image_batch = image_batch.cuda()
        label_batch = label_batch.cuda()
        outputs = model(image_batch)
        loss = Loss1(outputs, label_batch)

        loss_train_sum += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if (i_batch + 1) % 20 == 0:
            print(
                'epoch={:d}/{:d}, iteration={:d}/{:d}, learning_rate={:06f}, Loss={:.6f}, Time={:.4f}'
                    .format(epoch_num, max_epoch - 1, i_batch + 1, len(TrainingLoader), lr_, loss.item(),
                            time.time() - st))

    os.makedirs(os.path.join(args.save_model_path, "train", systime))
    log_train_path = os.path.join(args.save_model_path, "train", systime, "pre_log_train.csv")
    with open(log_train_path, "a") as f:
        f.writelines(
            "{},{}\n".format(epoch_num, loss_train_sum.item() / len(TrainingLoader)))

    model.eval()
    save_test_path_each = os.path.join(args.save_model_path, "test", systime, str(epoch_num))
    os.makedirs(save_test_path_each, exist_ok=True)

    for idx, im_pack in enumerate(TestLoader):
        with torch.no_grad():
            im, index = im_pack
            im = im.cuda()
            outputs = model(im)

    outputs = outputs.cpu().numpy()
    im = im.cpu().numpy()

    for i in range(outputs.shape[1]):
        save_path = os.path.join(
            save_test_path_each,
            "{}_material_decomposition.npy".format(
                material_dict[i]))
        np.save(save_path, outputs[0, i, :, :])
    save_path = os.path.join(
        save_test_path_each,
        "{}_DECT.npy".format(
            material_dict[4]))
    np.save(save_path, im[0, 0, :, :])
    save_path = os.path.join(
        save_test_path_each,
        "{}_DECT.npy".format(
            material_dict[5]))
    np.save(save_path, im[0, 1, :, :])

    checkpoint(model, epoch_num, "model", "regular")
