from tqdm import tqdm
import shutil
import os

src_folder = '../../dataset/cars'
dest_folder = '../../modified_datasets/cars_flat'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

num_kept = 0

for folder in tqdm(os.listdir(src_folder)):
    if '.' not in folder:
        for file in os.listdir(src_folder + '/' + folder):
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png"):
                src_img = src_folder + '/' + folder + '/' + file
                new_file_name = str(num_kept) + '.' + file.lower().split('.')[-1]
                dest_img = dest_folder + '/' + new_file_name
                shutil.copyfile(src_img, dest_img)

                src_label = src_folder + '/' + folder + '/' + file + '.json'
                if os.path.exists(src_label):
                    dest_label = dest_folder + '/' + new_file_name + '.json'
                    shutil.copyfile(src_label, dest_label)

                num_kept += 1
            elif not file.lower().endswith(".json"):
                print(file)

print('kept: ' + str(num_kept))
