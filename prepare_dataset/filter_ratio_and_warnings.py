from tqdm import tqdm
import shutil
import os
from PIL import Image
import warnings

src_folder = '../../modified_datasets/cars_flat'
dest_folder = '../../modified_datasets/cars_flat_ratio_warnings'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

num_kept = 0
num_removed = 0
num_corrupt_EXIF = 0

for file in tqdm(os.listdir(src_folder)):
    if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png"):
        src_img = src_folder + '/' + file
        dest_img = dest_folder + '/' + file

        src_label = src_folder + '/' + file + '.json'
        dest_label = dest_folder + '/' + file + '.json'

        with warnings.catch_warnings() as my_warning:
            warnings.simplefilter('error', UserWarning)
            try:
                img = Image.open(src_img)
                w, h = img.size
                if w < h:
                    print('removed invalid ratio')
                    num_removed += 1
                    continue
                shutil.copyfile(src_img, dest_img)
                if os.path.exists(src_label):
                    shutil.copyfile(src_label, dest_label)
                num_kept += 1
            except:
                print('removed invalid format')
                num_corrupt_EXIF += 1

print('Summary:')
print('removed corrupt_exif: ' + str(num_corrupt_EXIF))
print('removed: ' + str(num_removed))
print('kept: ' + str(num_kept))
