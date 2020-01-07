import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np

from Utilities import centeredText

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018.

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


def loadMetadata(filename, silent=False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


def normalize_image_transform(image_size, split, jitter):
    if jitter and split == 'train':
        normalize_image = transforms.Compose([
            transforms.Resize(240),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomCrop(image_size),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Well known ImageNet values
        ])
    else:
        normalize_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Well known ImageNet values
        ])

    return normalize_image


class ITrackerDataCPU(data.Dataset):
    def __init__(self, dataPath, imSize, gridSize, split='train', silent=False, jitter=True, color_space='YCbCr'):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize
        self.color_space = color_space

        print('Loading iTracker dataset...')
        metadata_file = os.path.join(dataPath, 'metadata.mat')

        if metadata_file is None or not os.path.isfile(metadata_file):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metadata_file)
        self.metadata = loadMetadata(metadata_file, silent)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metadata_file)

        self.normalize_image = normalize_image_transform(image_size=self.imSize, jitter=jitter, split=split)

        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        elif split == 'train':
            mask = self.metadata['labelTrain']
        elif split == 'all':
            mask = np.ones[len(self.metadata)]
        else:
            raise Exception('split should be test, val or train. The value of split was: {}'.format(split))

        self.indices = np.argwhere(mask)[:, 0]
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            # ToDo: Try YCbCr, HSV, LAB format
            # im = np.array(Image.open(path).convert('YCbCr'))
            im = Image.open(path).convert(self.color_space)
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
        return im

    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, real_index):
        index = self.indices[real_index]

        imFacePath = os.path.join(self.dataPath,
                                  '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index],
                                                               self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath,
                                  '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index],
                                                                  self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath,
                                  '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index],
                                                                   self.metadata['frameIndex'][index]))

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.normalize_image(imFace)
        imEyeL = self.normalize_image(imEyeL)
        imEyeR = self.normalize_image(imEyeR)

        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)
        frame = np.array([self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]])

        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index, :])

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze, frame, real_index

    def __len__(self):
        return len(self.indices)


class Dataset:
    def __init__(self, split, data, size, loader):
        self.split = split
        self.data = data
        self.size = size
        self.loader = loader


def load_data(split, path, image_size, grid_size, workers, batch_size, verbose, color_space, eval_boost):
    data = ITrackerDataCPU(path, image_size, grid_size, split=split, silent=not verbose, color_space=color_space)
    size = len(data.indices)
    shuffle = True if split == 'train' else False
    if eval_boost:
        batch_size = batch_size if split == 'train' else batch_size*2
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False)

    return Dataset(split, data, size, loader)


def load_all_data(path, image_size, grid_size, workers, batch_size, verbose, color_space='YCbCr'):
    print(centeredText('Loading Data'))
    eval_boost=False
    all_data = {
        # training data : model sees and learns from this data
        'train': load_data('train', path, image_size, grid_size, workers, batch_size, verbose, color_space, eval_boost),
        # validation data : model sees but never learns from this data
        'val': load_data('val', path, image_size, grid_size, workers, batch_size, verbose, color_space, eval_boost),
        # test data : model never sees or learns from this data
        'test': load_data('test', path, image_size, grid_size, workers, batch_size, verbose, color_space, eval_boost)
    }
    return all_data
