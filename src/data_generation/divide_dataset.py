import os
import shutil
import numpy as np
import sys
import colorama
# from Templates import *

TRAINCENT = 75
VALCENT = 15
TESTCENT = 100 - TRAINCENT - VALCENT


def progress_bar(current, total):
    total_progress = 50
    current_progress = int(50 * (current/total)) + 1

    output = 'Images Divided: {} \n[\033[91m'.format(current) + ('').join(['=' for _ in range(current_progress)])
    output += ('').join([' ' for _ in range(total_progress - current_progress)]) + '\033[0m]'

    return output                                

def getCurrentFileDirectory():
    return os.path.dirname(os.path.abspath(__file__))
def changeDir(folderPathString=None):
    if folderPathString is None:
        folderPathString = getCurrentFileDirectory()
    os.chdir(folderPathString)
# YOLO PATH
changeDir()

par_path = os.path.join("output")

# A folder with 2 folders of images and labels inside. From this, we extract
path = os.path.join(par_path)
# path = par_path
pimages = os.path.join(path, 'images')
plabels = os.path.join(path, 'lables')
images = os.listdir(pimages)
labels = os.listdir(plabels)
images.sort()
labels.sort()

# MAKING New YOLO folder AND ALL Folders WITHIN IT
newpath = os.path.join(par_path, 'Divided_Dataset')
if os.path.exists(newpath):
    shutil.rmtree(newpath)

os.mkdir(newpath)
for i in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(os.path.join(newpath, i), 'images'))
    os.makedirs(os.path.join(os.path.join(newpath, i), 'labels'))

new_ptrain = os.path.join(newpath, 'train')
new_pval = os.path.join(newpath, 'valid')
new_ptest = os.path.join(newpath, 'test')

# COPYING ALL FILES EXCEPT IMAGES AND LABELS
for file in os.listdir(par_path):
    if file != "data.yaml":
        continue
    file_path = os.path.join(par_path, file)
    if not os.path.isfile(file_path):
        continue

    shutil.copy(file_path, newpath)

# COPYING FILES INTO TRAIN, VALIDATION, TEST FOLDERS
TOTAL_IMAGES = NUMFILES = len(os.listdir(pimages))
perm = np.random.permutation(NUMFILES)
prev = 0

ImageNumber = 0
NTRAIN = TOTAL_IMAGES * TRAINCENT // 100
NVAL = TOTAL_IMAGES * VALCENT // 100
NTEST = TOTAL_IMAGES - NTRAIN - NVAL

os.system('cls' if os.name == 'nt' else 'clear')
for fol in ['train', 'valid', 'test']:
    if fol == 'train': num = NTRAIN
    elif fol == 'valid': num = NVAL
    elif fol == 'test': num = NTEST

    whereto = os.path.join(newpath, fol)
    for idx in perm[prev : prev + num]:
        shutil.move(os.path.join(pimages, images[idx]), os.path.join(whereto, 'images'))
        shutil.copy(os.path.join(plabels, labels[idx]), os.path.join(whereto, 'labels'))

        ImageNumber += 1
        if ImageNumber % 100 == 0:

            print(colorama.Cursor.POS(0, 1) + progress_bar(ImageNumber, NUMFILES))
    prev += num
