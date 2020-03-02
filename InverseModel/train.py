import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils, models
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR


np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

# label index in the poke data array (29, 1)
# index 0 is the poke index corresponding to starting image
stx, sty, edx, edy = 1, 2, 3, 4 # ee pos of start and end poke
obx, oby, qt1, qt2, qt3, qt4 = 5, 6, 7, 8, 9, 10 # obj pose before poke
js1, js2, js3, js4, js5, js6 = 11, 12, 13, 14, 15, 16 # jpos before poke
je1, je2, je3, je4, je5, je6 = 17, 18, 19, 20, 21, 22 # jpos after poke
sr, stc, edr, edc, obr, obc = 23, 24, 25, 26, 27, 28 # row and col locations in image
pang, plen = 29, 30 # poke angle and poke length


# default data transform
default_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                           std=[0.5, 0.5, 0.5])])

class PokeDataset(Dataset):

    def __init__(self, dirname, start_label, end_label, size=None, transform=None):
        self.dirname = dirname
        self.data = np.loadtxt(dirname + '.txt')
        self.transform = transform
        self.size = size
        self.start_label = start_label
        self.end_label = end_label

    def __len__(self):
        if self.size: # in cases where I want to define size
            return self.size
        else:
            return len(self.data)

    def __getitem__(self, idx):
        '''idx should be a single value, not list'''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx == -1:
            idx = self.__len__() - 1 # a hack to prevent accessing img1 out of range
        if idx > self.__len__()-1:
            raise ValueError('idx must be smaller than the len-1\
                                in order to get two images')

        img1_name = os.path.join(self.dirname, str(int(self.data[idx, 0]))) + '.png'
        # fix the last idx using identity poke (i.e. zero poke)
        if idx == self.__len__()-1:
            img2_name = img1_name
            poke = np.zeros((self.end_label-self.start_label+1,), dtype='float32')
        else:
            img2_name = os.path.join(self.dirname, str(int(self.data[idx, 0])+1)) + '.png'
            poke = np.float32(self.data[idx, self.start_label:self.end_label+1])

        poke = np.array(poke)
        img1 = imread(img1_name)
        img2 = imread(img2_name)
        img = np.vstack([img1, img2])

        if self.transform:
            img = self.transform(img)

        sample = {'img': img,
                  'poke': poke}
        return sample



def run_experiment(experiment_tag, seed, bsize, lr, num_epochs, nwork,
                    train_num, valid_num, use_pretrained):
    # set seed
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)

    # decide training objective: predictions in world, joint, or pixel space
    if experiment_tag == 'world':
        start_label, end_label = stx, oby
    elif experiment_tag == 'joint':
        start_label, end_label = js1, je6
    elif experiment_tag == 'pixel':
        start_label, end_label = sr, obc
    elif experiment_tag == 'wpoke':
        start_label, end_label = pang, plen
    else:
        raise Exception("experiment_tag has to be 'world', 'joint', 'pixel', or 'wpoke'")

    # build model
    print()
    if use_pretrained:
        print('Using pretrained weights...')
    else:
        print('Not using pretrained weights')
    model = models.resnet18(pretrained=use_pretrained)
    model.fc = nn.Linear(512, end_label-start_label+1)

    # specify data folders
    train_dirs = ['data/image_86', 'data/image_87', 'data/image_88']#['data/image_51', 'data/image_53','data/image_54', 'data/image_56', 'data/image_57']

    valid_dirs = ['data/image_85']
    data_dirs = {'train': train_dirs, 'val': valid_dirs}
    train_num_per_dir = train_num
    valid_num_per_dir = valid_num
    # data transform
    data_transforms = default_transform
    # build DataLoader
    # concatenate train sources into a single dataset
    list_of_train_sets = []
    for data_dir in train_dirs:
        pokeset = PokeDataset(data_dir, start_label, end_label, size=train_num_per_dir, transform=data_transforms)
        list_of_train_sets.append(pokeset)
    train_sets = ConcatDataset(list_of_train_sets)
    train_loader = DataLoader(train_sets, batch_size=bsize, shuffle=True, num_workers=nwork)
    # valid sources
    list_of_valid_sets = []
    for data_dir in valid_dirs:
        pokeset = PokeDataset(data_dir, start_label, end_label, size=valid_num_per_dir, transform=data_transforms)
        list_of_valid_sets.append(pokeset)
    valid_sets = ConcatDataset(list_of_valid_sets)
    valid_loader = DataLoader(valid_sets, batch_size=bsize, shuffle=False, num_workers=nwork)
    dataloaders = {'train': train_loader, 'val': valid_loader}

    # write to tensorboard images and model graphs
    writer = SummaryWriter('runs/run'+str(seed))
    dataiter = iter(train_loader)
    images = dataiter.next()['img'][:50]
    img_grid = utils.make_grid(images)
    writer.add_image('pokes', img_grid)
    writer.add_graph(model, images)
    writer.close()

    # start to run experiments
    model = model.float().cuda()
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    # record time
    since = time.time()
    last_time = since
    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 1e10 # some big number
    running_losses = {}
    print()
    print('batch size ', bsize)
    print('learning rate', lr)
    print('train sets ', train_dirs)
    print('valid sets ', valid_dirs)
    print('training num ', len(train_dirs) * train_num_per_dir)
    print('validation num ', len(valid_dirs) * valid_num_per_dir)
    print('')

    # running through epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        for param_group in optimizer.param_groups:
            print('learning rate: {:.0e}'.format(param_group['lr'])) # print curr learning rate
        for phase in ['train', 'val']:
            running_losses[phase] = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # iterate over data
            for batch_iter, batched_sample in enumerate(dataloaders[phase]):
                inputs = batched_sample['img']
                labels = batched_sample['poke']
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # L1 loss
                    loss = torch.abs(outputs - labels).mean()
                    # criterion = nn.SmoothL1Loss()
                    # loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # log statistics
                running_losses[phase] += loss.item() * inputs.size(0)
                if phase == 'train':
                    if batch_iter % 10 == 0:
                        writer.add_scalar('training_loss', loss.item(),
                            epoch*len(train_loader) + batch_iter)
                            # len(train_loader) gives how many batches are there in a loader
                else:
                    writer.add_scalar('valid_loss', loss.item(),
                        epoch*len(valid_loader) + batch_iter)

        # print training losses
        train_loss = running_losses['train'] / (len(train_dirs)*train_num_per_dir)
        print('Training Loss: {:.4f}'.format(train_loss))
        # print validation losses
        valid_loss = running_losses['val'] / (len(valid_dirs)*valid_num_per_dir)
        print('Validation Loss: {:.4f}'.format(valid_loss))

        # update learning rate with scheduler
        scheduler.step()

        # record weights
        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print('updating best weights...')

        # print running time
        curr_time = time.time()
        epoch_time = curr_time - last_time
        print('Current epoch time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
        time_elapsed = curr_time - since
        print('Training so far takes in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                            time_elapsed % 60))
        last_time = curr_time
        print()

    # save best model weights
    print("saving model's learned parameters...")
    save_path = 'weights/run' + str(seed) + '_' + experiment_tag + '_bs' + str(bsize) \
                + '_lr' + str(lr) + '_ep' + str(num_epochs) +'.pth'
    torch.save(model.state_dict(), save_path)
    print("model saved to ", save_path)
    print()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # omit argument to produce no rendering
    parser.add_argument('--tag', required=True, help="experiment tag: 'world', 'joint', 'pixel'")
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--bsize', type=int, default=500, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--epoch', type=int, default=4, help='num of epochs')
    parser.add_argument('--nwork', type=int, default=8, help='num of workers')
    parser.add_argument('--train_size', type=int, default=70000, help='num of data from each train dir')
    parser.add_argument('--valid_size', type=int, default=20000, help='num of data from each valid dir')
    parser.add_argument('--use_init', action='store_true', help='use pretrained weights')
    args = parser.parse_args()
    run_experiment(experiment_tag=args.tag, seed=args.seed, bsize=args.bsize, lr=args.lr,
        num_epochs=args.epoch, nwork=args.nwork, train_num=args.train_size,
        valid_num=args.valid_size, use_pretrained=args.use_init)
