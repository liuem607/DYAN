############################# Import Section #################################
## Imports related to PyTorch
import torch
from torch.autograd import Variable

## Generic imports
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from DyanOF import OFModel
from utils import getListOfFolders

from skimage import measure
from scipy.misc import imread, imresize
############################# Import Section #################################


# Hyper Parameters
FRA = 3 # if Kitti: FRA = 9
PRE = 1 # represents predicting 1 frame
N_FRAME = FRA+PRE
T = FRA
numOfPixels = 240*320 # if Kitti: 128*160

gpu_id = 1
opticalflow_ckpt_file = 'preTrainedModel/UCFModel.pth' # if Kitti: 'KittiModel.pth'

def loadOpticalFlowModel(ckpt_file):
	loadedcheckpoint = torch.load(ckpt_file)
	stateDict = loadedcheckpoint['state_dict']
	
	# load parameters
	Dtheta = stateDict['l1.theta'] 
	Drr    = stateDict['l1.rr']
	model = OFModel(Drr, Dtheta, FRA,PRE,gpu_id)
	model.cuda(gpu_id)
	
	return model

def warp(input,tensorFlow):
	torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
	torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2), input.size(3))
	torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
	torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

	tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda(gpu_id)
	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=input, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

##################### Only for Kitti dataset need to define: ##############

#def process_im(im, desired_sz=(128, 160)):
#    target_ds = float(desired_sz[0])/im.shape[0]
#    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
#    d = int((im.shape[1] - desired_sz[1]) / 2)
#    im = im[:, d:d+desired_sz[1]]
#    return im


#def SSIM(predi,pix):
#    pix = pix.astype(float)
#    predict = predi.astype(float)
#    ssim_score = measure.compare_ssim(pix, predict, win_size=11, data_range = 1.,multichannel=True,
#                                      gaussian_weights=True,sigma = 1.5,use_sample_covariance=False,
#                                      K1=0.01,K2=0.03)
                                      
#    return ssim_score

#mse = []
#ssim = []

############################################################################

## Load the model
ofmodel = loadOpticalFlowModel(opticalflow_ckpt_file)
ofSample = torch.FloatTensor(2, FRA, numOfPixels)

# set test list name:
testFolderFile = 'testlist01.txt'
# set test data directory:
rootDir = '/data/Abhishek/frames/'
# for UCF dataset:
testFoldeList = getListOfFolders(testFolderFile)[::10]
## if Kitti: use folderList instead of testFoldeList
## folderList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
## folderList.sort()

flowDir = '/home/abhishek/Workspace/UCF_Flows/Flows_ByName/'

for	numfo,folder in enumerate(testFoldeList):
	print("Started testing for - "+ folder)

	if not os.path.exists(os.path.join("Results", str(10*numfo+1))):
		os.makedirs(os.path.join("Results", str(10*numfo+1)))
	
	frames = [each for each in os.listdir(os.path.join(rootDir, folder)) if each.endswith(('.jpeg'))]
	frames.sort()

	path = os.path.join(rootDir,folder,frames[4])
	img = Image.open(path)
	original = np.array(img)/255.

	path = os.path.join(rootDir,folder,frames[3])
	img = Image.open(path)
	frame4 = np.array(img)/255.

	tensorinput = torch.from_numpy(frame4).type(torch.FloatTensor).permute(2,0,1).cuda(gpu_id).unsqueeze(0)
	
	for k in range(3):
		flow = np.load(os.path.join(flowDir,folder,str(k)+'.npy'))
		flow = np.transpose(flow,(2,0,1))
		ofSample[:,k,:] = torch.from_numpy(flow.reshape(2,numOfPixels)).type(torch.FloatTensor)
	
	ofinputData = ofSample.cuda(gpu_id)
	
	with torch.no_grad():	
		ofprediction = ofmodel.forward(Variable(ofinputData))[:,3,:].data.resize(2,240,320).unsqueeze(0)
	
	warpedPrediction = warp(tensorinput,ofprediction).squeeze(0).permute(1,2,0).cpu().numpy()
	warpedPrediction = np.clip(warpedPrediction, 0, 1.)


	plt.imsave(os.path.join("Results", str(10*numfo+1),'GTFrame-%04d' % (5)+'.png'), original)
	plt.imsave(os.path.join("Results", str(10*numfo+1),'PDFrame-%04d' % (5)+'.png'), warpedPrediction)
	plt.close()


##################### Testing script ONLY for Kitti dataset: ##############

#for folder in folderList:
#    print("Started testing for - "+ folder)
    
#    frames = [each for each in os.listdir(os.path.join(rootDir, folder)) if each.endswith(('.jpeg','png'))]
#    frames.sort()
    
#    for i in range(0,len(frames)-11,10):
#        imgname = os.path.join(rootDir,folder,frames[i+10])
#        img = Image.open(imgname)
#        original = process_im(np.array(img))/255.
        
#        path = os.path.join(rootDir,folder,frames[i+9])
#        img = Image.open(path)
#        frame10 = process_im(np.array(img))/255.
        
#        tensorinput = torch.from_numpy(frame10).type(torch.FloatTensor).permute(2,0,1).cuda(gpu_id).unsqueeze(0)
        
#        for k in range(9):
#            flow = np.load(os.path.join(flowDir,folder,str(k)+'.npy'))
#            flow = np.transpose(flow,(2,0,1))
#            ofSample[:,k,:] = torch.from_numpy(flow.reshape(2,numOfPixels)).type(torch.FloatTensor)
    
#        ofinputData = ofSample.cuda(gpu_id)
        
#        with torch.no_grad():
#            ofprediction = ofmodel.forward(Variable(ofinputData))[:,9,:].data.resize(2,128,160).unsqueeze(0)

#        warpedPrediction = warp(tensorinput,ofprediction).squeeze(0).permute(1,2,0).cpu().numpy()
#        img_back = np.clip(warpedPrediction, 0, 1.)
#        meanserror = np.mean( (img_back - original) ** 2 )
#        mse.append(meanserror)
#        ssim.append(SSIM(original, img_back))

#print("MSE : ", np.mean(np.array(mse)))
#print("SSIMs : ", np.mean(np.array(ssim)))

############################################################################
