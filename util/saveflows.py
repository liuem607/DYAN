import os 
import numpy as np
import pyflow
import pandas as pd
from PIL import Image

def getListOfFolders(File):
	data = pd.read_csv(File, sep=" ", header=None)[0]
	data = data.str.split('/',expand=True)[1]
	data = data.str.rstrip(".avi").values.tolist()

	return data


alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0

testFolderFile = 'trainlist01.txt'
trainFoldeList = getListOfFolders(testFolderFile)[::10]
saveDir = '/data/Abhishek/sample'
rootDir = '/data/Abhishek/frames'

print(len(trainFoldeList))

for foldernum,folder in enumerate(trainFoldeList):
	frames = [each for each in os.listdir(os.path.join(rootDir,folder)) if each.endswith(('.jpg','.jpeg'))]
	nFrames = len(frames)
	frames.sort()
	print(folder, foldernum+1)
	for framenum in range(0,10):
		imgname = os.path.join(rootDir,folder,frames[framenum])
		img1 = np.array(Image.open(imgname))/255.
		
		imgname = os.path.join(rootDir,folder,frames[framenum+1])
		img2 = np.array(Image.open(imgname))/255.
		
		u, v,_ = pyflow.coarse2fine_flow( img2, img1, alpha, ratio, minWidth, 
							nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
		flow = np.concatenate((u[..., None], v[..., None]), axis=2)
		if not os.path.exists(os.path.join(saveDir, folder)):
			os.makedirs(os.path.join(saveDir, folder))
		np.save(os.path.join(saveDir, folder,str(framenum)+'.npy'), flow)