from tqdm import tqdm
import shutil
import os

src_folder = 'E:/carswithcolors/trainA'
dest_folder = 'E:/carswithcolors/images_with_labels'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

num_kept = 0

for folder in tqdm(os.listdir(src_folder)):
    if '.' not in folder:
        for file in os.listdir(src_folder + '/' + folder):
            if file.endswith(".jpg"):
                src_img = src_folder + '/' + folder + '/' + file
                dest_img = dest_folder + '/' + file
                shutil.copyfile(src_img, dest_img)

                src_label = src_folder + '/' + folder + '/' + file + '.json'
                dest_label = dest_folder + '/' + file + '.json'
                shutil.copyfile(src_label, dest_label)

                num_kept += 1

print('kept: ' + str(num_kept))
