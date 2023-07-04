import torch.backends.cudnn as cudnn
import argparse
import os
import time
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from datetime import datetime
import random
import glob
from networks.DECT_CNN import *
from data_loader import DataLoader_SECT


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
                    default='../Neighbor2Neighbor-main(explain)/datasets/DECT_data_denoised/train',
                    help='root dir for train data')
parser.add_argument('--test_data_dir', type=str,
                    default='../Neighbor2Neighbor-main(explain)/datasets/DECT_data_denoised/test',
                    help='root dir for test data')
parser.add_argument('--log_name', type=str,
                    default='material_decomposition', help='network function')
parser.add_argument('--save_model_path', type=str,
                    default='./SECT_DECT_results', help='dir for network weight')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
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

TrainingDataset = DataLoader_SECT(args.train_data_dir)
TestDataset = DataLoader_SECT(args.test_data_dir)

TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

TestLoader = DataLoader(dataset=TestDataset,
                        num_workers=8,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=True)

model = UNet(in_channels=1, out_channels=2).cuda()

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
        image_batch = ima
        image_batch = image_batch.cuda()
        input_image = torch.sum(image_batch * 0.5, dim=1, keepdim=True)
        label = image_batch
        outputs = model(input_image)
        loss = Loss1(outputs, label)
        loss_train_sum += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if (i_batch + 1) % 20 == 0:
            print()
            print(
                'epoch={:d}/{:d}, iteration={:d}/{:d}, learning_rate={:06f}, Loss={:.6f}, Time={:.4f}'
                    .format(epoch_num, max_epoch - 1, i_batch + 1, len(TrainingLoader), lr_, loss.item(),
                            time.time() - st))

    os.makedirs(os.path.join(args.save_model_path, "train", systime), exist_ok=True)
    log_train_path = os.path.join(args.save_model_path, "train", systime, "log_train.csv")
    with open(log_train_path, "a") as f:
        f.writelines(
            "{},{}\n".format(epoch_num, loss_train_sum.item() / len(TrainingLoader)))

    model.eval()
    loss_test_sum = 0

    for idx, im_pack in enumerate(TestLoader):
        with torch.no_grad():
            im = im_pack.cuda()
            inp = torch.sum(im * 0.5, dim=1, keepdim=True)
            outputs = model(inp)
            loss = Loss1(outputs, im)

            loss_test_sum += loss

    os.makedirs(os.path.join(args.save_model_path, "test", systime), exist_ok=True)
    log_path = os.path.join(args.save_model_path, "test", systime, "log.csv")
    with open(log_path, "a") as f:
        f.writelines(
            "{},{}\n".format(epoch_num, loss_test_sum.item() / len(TestLoader)))

    checkpoint(model, epoch_num, "model", "regular")
