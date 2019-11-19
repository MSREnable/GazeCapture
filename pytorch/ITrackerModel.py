import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

'''
Pytorch model for the iTracker.
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

class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    # output = (input-k+2p)/s + 1
    # ZeroPad = (k-1)/2
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            # 3C x 224H x 224W
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  # CONV-1
            # 96C x 54H x 54W
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 96C x 26H x 26W
            nn.ReLU(inplace=True),
            
            # 96C x 26H x 26W
            nn.BatchNorm2d(96),
            nn.Dropout2d(0.1),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),  # CONV-2
            # 256C x 26H x 26W
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256C x 12H x 12W
            nn.ReLU(inplace=True),

            # 256C x 12H x 12W
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # CONV-3
            # 384C x 12H x 12W
            nn.ReLU(inplace=True),

            # 384C x 12H x 12W
            nn.BatchNorm2d(384),
            nn.Dropout2d(0.1),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),  # CONV-4
            # 64C x 12H x 12W
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        # 64C x 12H x 12W
        x = x.view(x.size(0), -1)
        # 9216 (64x12x12)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            # 9216 (64x12x12)
            nn.Dropout(0.1),
            nn.Linear(12 * 12 * 64, 128),  # FC-F1
            # 256
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # FC-F2
            # 128
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # 3C x 224H x 224W
        x = self.conv(x)
        # 9216 (64x12x12)
        x = self.fc(x)
        # 64
        return x


class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            # 625 (25x25)
            nn.Linear(gridSize * gridSize, 256),  # FC-FG1
            # 256
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.1),
            nn.Linear(256, 128),  # FC-FG2
            # 128
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 25x25
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # 128
        return x


class ITrackerModel(nn.Module):

    def __init__(self):
        super(ITrackerModel, self).__init__()
        # 3Cx224Hx224W --> 9216 (64x12x12)
        self.eyeModel = ItrackerImageModel()
        # 3Cx224Hx224W --> 64
        self.faceModel = FaceImageModel()
        # 1Cx25Hx25W --> 128
        self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * 12 * 12 * 64, 128),  # FC-E1
            nn.ReLU(inplace=True),
        )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128 + 64 + 128, 128),  # FC1
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.1),
            nn.Linear(128, 2),  # FC2
        )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)

        return x
