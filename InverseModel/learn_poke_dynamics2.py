import os,sys
import os.path 
from os import path, listdir
from os.path import isfile, join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import atexit
import time
import datetime
import pickle
import signal
import torch
import torchvision
from torch import nn
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, models
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import linecache
from PIL import Image
from matplotlib.image import imread
from torch.optim.lr_scheduler import StepLR

# still inverse dynamics
# img1 -> resnet -> phi(img1) -> 3 layers -> action
# img2 -> resnet -> phi(img2) 
class Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, 
		des_path = '280kpokes/image_85', 
		length = 0):
		'Initialization'
		self.length = length
		self.transform = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize(mean=[0.5, 0.5, 0.5],
							std=[0.5, 0.5, 0.5])])
		# self.action_mean = np.array([0,0.08])
		# self.action_std = np.array([1.1*np.pi, 0.025])
		self.action_normalization = lambda x: (x - self.action_mean)/self.action_std
		self.des_path = des_path
		self.data = np.loadtxt(des_path + '.txt')
		# import pdb; pdb.set_trace()

	def __len__(self):
		'Denotes the total number of samples'
		return self.length

	def __getitem__(self, index):
		'Generates one sample of data'
		# Load data
		img1 = imread( os.path.join(self.des_path, str(index) + '.png') )
		img2 = imread( os.path.join(self.des_path, str(index + 1) + '.png') )
		img1 = self.transform(img1)
		img2 = self.transform(img2)
		act = self.data[index, 1:5]
		act = np.float32(act)

		sample = {'img1': img1,
				  'img2': img2,
				  'act': act}
		return sample


class InverseModel(nn.Module):
	def __init__(self,
		hidden_dim1,
		hidden_dim2,
		hidden_dim3,
		feature_dim,
		output_dim):
		super(InverseModel, self).__init__()
		self.encoder = models.resnet18(pretrained=True)
		self.encoder.fc = nn.Linear(512, feature_dim)
		self.final_layers = nn.Sequential(
			nn.Linear(feature_dim*2, hidden_dim1),
			nn.ReLU(True),
			nn.Linear(hidden_dim1, hidden_dim2),
			nn.ReLU(True),
			nn.Linear(hidden_dim2, hidden_dim3),
			nn.ReLU(True),
			nn.Linear(hidden_dim3, output_dim),
			# nn.Tanh()
		)

	def forward(self, img1, img2):
		phi_img1 = self.encoder(img1)
		phi_img2 = self.encoder(img2)
		action = self.final_layers(torch.cat((phi_img1, phi_img2),1))
		return action

class TrainInverseModel:
	def __init__(self):
		self.t = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
		self.python_script_name = 'learn_poke_dynamics2'
		feature_dim = 100
		self.jnt_dim = 4

		self.learning_rate = 1e-4
		self.model = InverseModel(200,200,200,feature_dim,self.jnt_dim).float().cuda()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
		self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)
		self.alpha = 0.1
		self.data_length = 64000

	def train(self):
		# CUDA for PyTorch
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0" if use_cuda else "cpu")

		# Parameters
		params = {'batch_size': 1,
		          'shuffle': True,
		          'num_workers': 6}
		max_epochs = 1000

		# training generators
		train_dirs = ['280kpokes/image_85', '280kpokes/image_86']
		data_length = self.data_length#70730 - 1
		train_set_list = []
		for data_dir in train_dirs:
			train_set_list.append( Dataset(data_dir, data_length) )
		training_set = ConcatDataset(train_set_list)
		training_generator = DataLoader(training_set, **params)

		# validation generators
		valid_dirs = ['280kpokes/image_87']
		valid_set_list = []
		for data_dir in valid_dirs:
			valid_set_list.append( Dataset(data_dir, data_length) )
		valid_set = ConcatDataset(valid_set_list)
		valid_generator = DataLoader(valid_set, **params)

		# # write to tensorboard images and model graphs
		# seed = 1
		# writer = SummaryWriter('runs/run'+str(seed))
		# dataiter = iter(training_generator)
		# images,_,_ = dataiter.next()
		# img_grid = utils.make_grid(images)
		# writer.add_image('pokes', img_grid)
		# writer.add_graph(self.model, images)
		# writer.close()

		action_loss = nn.CrossEntropyLoss()
		mse_loss = nn.MSELoss()

		self.save_path = os.path.join('.',self.python_script_name + '_data', 'exp_' + self.t)
		os.makedirs(self.save_path)
		fout11 = open(self.save_path + '/training_inverse_loss_data.txt', 'w')
		fout21 = open(self.save_path + '/validation_inverse_loss_data.txt', 'w')

		running_losses = {}
		start_time = time.time()
		# Loop over epochs
		for epoch in range(max_epochs):

			running_losses['train_inverse_loss'] = 0.0
			running_losses['val_inverse_loss'] = 0.0
		    # Training
			i = 0
			for sample_batch in training_generator:
				img1_batch = sample_batch['img1']
				img2_batch = sample_batch['img2']
				act_batch = sample_batch['act']
				# import pdb;pdb.set_trace()
		        # Transfer to GPU
				img1_batch = img1_batch.to(device)
				img2_batch = img2_batch.to(device)
				act_batch = act_batch.to(device)
				# forward
				act_hat = self.model(img1_batch,img2_batch)
				# import pdb; pdb.set_trace()
				loss_inverse = mse_loss(act_hat, act_batch)
				loss = loss_inverse
				# backward
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				# running losses
				running_losses['train_inverse_loss'] += loss_inverse.item() * img1_batch.size(0)
				# # write to tensorboard
				# writer.add_scaler('training_loss', loss.item(), epoch*len(training_generator) + i) 
				i += 1

			i = 0
			for sample_batch in valid_generator:
				img1_batch = sample_batch['img1']
				img2_batch = sample_batch['img2']
				act_batch = sample_batch['act']
				img1_batch = img1_batch.to(device)
				img2_batch = img2_batch.to(device)
				act_batch = act_batch.to(device)
				act_hat = self.model(img1_batch, img2_batch)
				loss_inverse = mse_loss(act_hat, act_batch)
				# running losses
				running_losses['val_inverse_loss'] += loss_inverse.item() * img1_batch.size(0)
				# # write to tensorboard
				# writer.add_scaler('valid_loss', loss.item(), epoch*len(valid_generator) + i)
				i += 1

			# print time
			print("--- %s seconds ---" % (time.time() - start_time))
			# print training losses
			train_inverse_loss = running_losses['train_inverse_loss'] / (len(train_dirs)*data_length)
			print('Training Loss: {:.4f}'.format(train_inverse_loss))
			# print validation losses
			valid_inverse_loss = running_losses['val_inverse_loss'] / (len(valid_dirs)*data_length)
			print('Validation Loss: {:.4f}'.format(valid_inverse_loss))

			# update learning rate with scheduler
			self.scheduler.step()

			# save losses
			fout11.write('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, max_epochs, train_inverse_loss) + '\n')
			fout11.flush()
			fout21.write('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, max_epochs, valid_inverse_loss) + '\n')
			fout21.flush()

			# save model
			torch.save(self.model.state_dict(), os.path.join(self.python_script_name + '_data', 'exp_' + self.t, 'model_' + str(epoch) + '.pth') )
		# torch.save(self.model.state_dict(), os.path.join(self.python_script_name + '_data', 'exp_' + self.t, 'model.pth') )
		fout11.close()
		fout21.close()

	def test(self, 
		load_model=1,
		model_path='learn_poke_dynamics2_data/exp_2020-03-05-00-05-33',
		model_number='model_2.pth'):
		if load_model:
			self.model.load_state_dict(torch.load( os.path.join(model_path, model_number) ))
		use_cuda = torch.cuda.is_available()
		device = torch.device("cuda:0" if use_cuda else "cpu")

		# Parameters
		params = {'batch_size': 1,
		          'shuffle': False,
		          'num_workers': 6}

		# validation generators
		# test_dirs = ['280kpokes/image_88']
		data_file_str = 'image_88'
		test_dirs = ['280kpokes/'+ data_file_str]
		test_set_list = []
		data_length = self.data_length
		for data_dir in test_dirs:
			test_set_list.append( Dataset(data_dir, data_length) )
		test_set = ConcatDataset(test_set_list)
		test_generator = DataLoader(test_set, **params)

		action_loss = nn.CrossEntropyLoss()
		mse_loss = nn.MSELoss()

		self.save_path = os.path.join('.',self.python_script_name + '_data', 'exp_' + self.t)
		os.makedirs(self.save_path)
		# fout = open(self.save_path + '/pd_image_88.txt', 'w')

		running_losses = {}
		start_time = time.time()
		# Loop over epochs

		running_losses['test'] = 0.0
	    # Training
		save_data = np.zeros((1,self.jnt_dim))
		save_true_data = np.zeros((1,self.jnt_dim))
		i = 0
		for sample_batch in test_generator:
			img1_batch = sample_batch['img1']
			img2_batch = sample_batch['img2']
			# print(img_batch.detach().cpu().numpy()[9])
			# import pdb; pdb.set_trace()
			act_batch = sample_batch['act']
	        # Transfer to GPU
			img1_batch = img1_batch.to(device)
			img2_batch = img2_batch.to(device)
			act_batch = act_batch.to(device)
			# Model computations
			# forward
			act_hat = self.model(img1_batch, img2_batch)
			# import pdb; pdb.set_trace()
			loss = mse_loss(act_hat, act_batch) 
			# running losses
			running_losses['test'] += loss.item() * img1_batch.size(0)
			# # write to tensorboard
			# writer.add_scaler('training_loss', loss.item(), epoch*len(training_generator) + i) 
			save_data_point = act_hat.cpu().detach().numpy()
			save_data = np.concatenate((save_data, save_data_point), axis=0)
			save_true_data_point = act_batch.cpu().detach().numpy()
			save_true_data = np.concatenate((save_true_data, save_true_data_point), axis=0)
			i += 1
		test_loss = running_losses['test'] / (len(test_dirs)*data_length)
		print('Test Loss: {:.4f}'.format(test_loss))
		save_data = np.delete(save_data,0,0)
		save_true_data = np.delete(save_true_data, 0, 0)
		# save_data = save_data * self.action_std + self.action_mean
		save_data_idx = np.asarray(range(data_length))
		save_data = np.concatenate((save_data_idx.reshape(-1,1), save_data), axis=1)
		save_true_data = np.concatenate((save_data_idx.reshape(-1,1), save_true_data), axis=1)
		np.savetxt(self.save_path + '/pd_' + data_file_str + '_lpd2.txt', save_data)
		np.savetxt(self.save_path + '/true_' + data_file_str + '.txt', save_true_data)
		# fout.close()

if __name__ == '__main__':
	## train
	# a = TrainInverseModel()
	# a.train()

	a = TrainInverseModel()
	a.test(model_path='learn_poke_dynamics2_data/exp_2020-03-05-00-05-33/',model_number='model_5.pth')