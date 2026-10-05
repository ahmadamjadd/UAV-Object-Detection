import os
import shutil
import numpy as np
import sys
from functools import reduce
# from Templates import *
import colorama

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
letters = [char for char in [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]]

# A folder with 2 folders of images and labels inside. From this, we extract
path = os.path.join(par_path)
# path = par_path
pimages = os.path.join(path, 'letter_images')
plabels = os.path.join(path, 'letter_lables')

# MAKING New YOLO folder AND ALL Folders WITHIN IT
newpath = os.path.join(par_path, 'Divided_Letter_Dataset')
if os.path.exists(newpath):
    shutil.rmtree(newpath)

os.mkdir(newpath)
for i in ['train', 'valid', 'test']:
    for letter in letters: 
        os.makedirs(os.path.join(os.path.join(newpath, i), letter))

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
ImageNumber = 0
NUMFILES = reduce(lambda x, y: x + y, [len(os.listdir(os.path.join(pimages, letter))) for letter in letters])
print(colorama.Cursor.POS(0, 1) + f"Total Images: {NUMFILES}              ")

os.system('cls' if os.name == 'nt' else 'clear')
for letter in letters: 
    letterimages = os.path.join(pimages, letter)
    TOTAL_IMAGES = len(os.listdir(letterimages))

    images = os.listdir(letterimages)
    images.sort()
    
    perm = np.random.permutation(TOTAL_IMAGES)
    prev = 0

    NTRAIN = TOTAL_IMAGES * TRAINCENT // 100
    NVAL = TOTAL_IMAGES * VALCENT // 100
    NTEST = TOTAL_IMAGES - NTRAIN - NVAL
    for fol in ['train', 'valid', 'test']:
        if fol == 'train': num = NTRAIN
        elif fol == 'valid': num = NVAL
        elif fol == 'test': num = NTEST

        whereto = os.path.join(newpath, fol, letter)
        for idx in perm[prev : prev + num]:
            shutil.move(os.path.join(letterimages, images[idx]), os.path.join(whereto))

            ImageNumber += 1
            if ImageNumber % 100 == 0:
                print(colorama.Cursor.POS(0, 2) + progress_bar(ImageNumber, NUMFILES))
        prev += num
