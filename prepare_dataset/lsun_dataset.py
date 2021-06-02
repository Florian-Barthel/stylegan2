import lmdb
import os
from os.path import exists, join


def export_images(db_path, out_dir, flat=False, limit=-1):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            if not flat:
                image_out_dir = join(out_dir, '/'.join(str(key[:6])))
            else:
                image_out_dir = out_dir
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, str(key) + '.webp')
            with open(image_out_path, 'w') as fp:
                fp.write(str(val))
            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')

export_images('../../lsun-cars/car', '../../lsun-cars/images', True)
