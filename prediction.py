import tensorflow as tf
from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import argparse
import numpy as np
from csbdeep.models import CARE
import os
base_path = os.path.abspath('')


def main():
    factor, clip = 800., 2000.    
    
    def relu_advanced(x):
        return K.relu(x, max_value=63 * factor)

    parser = argparse.ArgumentParser(description='Use the script to translate VIIRS to DMSP-like data. Shape of the file should be (height, width, 1) where 1 is the number of channels.')
    parser.add_argument('path', type=str, help='Path to the file you want to perform translation on. '
                                               'Note, that if you want to process .envi file you need to use'
                                               " --header argument and set the path to the file's header. "
                                               "Otherwise your file is considered to be in .npz format")
    parser.add_argument('--header', type=str, default=None)
    parser.add_argument('--split', type=int, default=1, help='Number of tiles to split the imagery into.'
                                                             ' In most cases the optimal split will be'
                                                             ' found automatically but if it fails you'
                                                             ' can use this parameter to set it up manually')

    args = parser.parse_args()
    if args.header is None:
        img = np.load(base_path + '/' + args.path)
        img = img['arr_0']
    else:
        import spectral.io.envi as envi
        # from spectral import *
        img = envi.open(base_path + '/' + args.header,
                        base_path + '/' + args.path)
        img = img.load()
        img = np.clip(img, 0, clip)
    img = np.moveaxis(img, -1, 0)
    get_custom_objects().update({'linear': Activation(relu_advanced)})
    model = CARE(config=None, name='model', basedir=base_path)
    axes = 'CYX'
    restored = model.predict(img, axes, n_tiles=(1, args.split, args.split), normalizer=None)
    restored = restored.reshape(*restored.shape[1:])
    restored = restored / factor
    restored = np.around(restored, 0)
    np.savez(base_path + '/result.npz', restored)
    print('Finished restoration!')


if __name__ == '__main__':
    main()
