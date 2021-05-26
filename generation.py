import tensorflow as tf
import tensorflow_addons as tfa
from skimage.util import view_as_windows
import argparse
import numpy as np
import os
from tqdm import tqdm
base_path = os.path.abspath('')


def split_and_predict(x, model, split=(8, 8, 1), overlap=1 / 3, cutoff=5, batch_size=24):
    def gen_restore_data(x_train_windows):
        def gen():
            for i in range(x_train_windows.shape[0]):
                for j in range(x_train_windows.shape[1]):
                    cnt = np.count_nonzero(x_train_windows[i, j, 0] > 0)
                    if cnt != 0:
                        yield x_train_windows[i, j, 0]

        return gen

    def get_window(a):
        a = np.squeeze(a)
        std = np.max(a.shape) / 4
        center = (np.array((a.shape[0], a.shape[1]))[np.newaxis, ...] - 1) / 2
        a = np.zeros_like(a)
        ind = np.array(np.where(a == 0))
        ind = np.moveaxis(ind, 0, -1).reshape((*a.shape, 2))
        a = np.linalg.norm(ind - center, ord=2, axis=-1)
        a = np.exp(-(np.power(a, 2) / (2 * std ** 2))) / (std * (2 * np.pi) ** (1 / 2))
        return a / a.max()

    def perform_transfer(x, patch_size=256, step=64, batch_size=batch_size):
        orig_shape = x.shape
        pad_values = (patch_size - 1 + step, patch_size - 1 + step)
        x = np.pad(x, ((0, pad_values[0]), (0, pad_values[1]), (0, 0)))
        y = np.zeros_like(x)
        y_coef = np.zeros_like(x)
        window_coef = np.full((patch_size, patch_size, 1), 10E-5)
        window_coef[..., 0] = get_window(window_coef[...])
        x_train_windows = view_as_windows(x, (patch_size, patch_size, 1), step=step)

        dataset = tf.data.Dataset.from_generator(
            gen_restore_data(x_train_windows),
            output_signature=tf.TensorSpec(shape=(patch_size, patch_size, 1)))
        dataset = dataset.batch(batch_size)
        try:
            res = model.predict(dataset, verbose=0)
        except ValueError:
            return y[:orig_shape[0], :orig_shape[1]]
        counter = 0
        y_windows = view_as_windows(y, (patch_size, patch_size, 1), step=step)
        y_coef_windows = view_as_windows(y_coef, (patch_size, patch_size, 1), step=step)
        for i in range(y_windows.shape[0]):
            for j in range(y_windows.shape[1]):
                cnt = np.count_nonzero(x_train_windows[i, j, 0] > 0)
                if cnt != 0:
                    y_windows[i, j, 0] += res[counter] * window_coef
                    counter += 1
                y_coef_windows[i, j, 0] += window_coef
        y = y / (y_coef + 10E-5)
        return y[:orig_shape[0], :orig_shape[1]]

    step = np.array(x.shape) // np.array(split)
    overlap = step * overlap
    overlap = overlap.astype(int)
    step = step.astype(int)
    orig_shape = x.shape
    pad_values = (step[0] + overlap[0], step[1] + overlap[1])
    x = np.pad(x, ((cutoff, pad_values[0]), (cutoff, pad_values[1]), (0, 0)))
    y = np.zeros_like(x)
    coefs = np.zeros_like(x)
    i_s_l = [i * step[0] for i in range(split[0] + 2)]
    j_s_l = [i * step[1] for i in range(split[1] + 2)]
    i_s_r = [i + overlap[0] for i in i_s_l[1:]]
    j_s_r = [i + overlap[1] for i in j_s_l[1:]]
    window_coef = get_window(np.zeros_like(x[:i_s_r[0], :j_s_r[0]]))[..., np.newaxis]
    window_aux = np.zeros_like(window_coef)
    window_aux[cutoff:-cutoff, cutoff:-cutoff] = 1
    np.multiply(window_coef, window_aux, out=window_coef)
    window_aux = None
    for i_l, i_r in tqdm(zip(i_s_l[:-1], i_s_r), total=len(i_s_r)):
        for j_l, j_r in zip(j_s_l[:-1], j_s_r):
            temp = perform_transfer(x[i_l: i_r, j_l: j_r])
            np.multiply(temp, window_coef, out=temp)
            np.add(y[i_l: i_r, j_l: j_r], temp, out=y[i_l: i_r, j_l: j_r])
            np.add(coefs[i_l: i_r, j_l: j_r], window_coef, coefs[i_l: i_r, j_l: j_r])
    np.add(coefs, 10E-5, out=coefs)
    np.divide(y, coefs, out=y)
    return y[cutoff: orig_shape[0] + cutoff, cutoff: orig_shape[1] + cutoff]


def main():
    clip = 2000.
    parser = argparse.ArgumentParser(
        description='Use the script to generate DMSP-like data from VIIRS data. Shape of the file should be (height, '
                    'width, 1) where 1 is the number of channels.')
    parser.add_argument('path', type=str, help='Path to the file you want to perform translation on. '
                                               'Note, that if you want to process .envi file you need to use'
                                               " --header argument and set the path to the file's header. "
                                               "Otherwise your file is considered to be in .npz format")
    parser.add_argument('--header', type=str, default=None)
    parser.add_argument('--split', type=int, default=8, help='Number of tiles to split the imagery into.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batches to be used in the prediction '
                                                                   'process.')
    parser.add_argument('--model', type=str, default='/model/model_MAE_depth_4_per_level_3.h5', help='Path to the model')

    args = parser.parse_args()
    if args.header is None:
        img = np.load(base_path + '/' + args.path)
        img = img['arr_0']
    else:
        import spectral.io.envi as envi
        img = envi.open(base_path + '/' + args.header,
                        base_path + '/' + args.path)
        img = img.load()
    img = np.clip(img, 0, clip)
    img = img / img.max()

    model = tf.keras.models.load_model(args.model)
    model.compile()

    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    res = split_and_predict(img, model, (int(args.split), int(args.split), 1), batch_size=args.batch_size)
    res = res * 63
    res = np.around(res, 0)
    np.savez(base_path + '/result.npz', res)
    print('Finished restoration!')


if __name__ == '__main__':
    main()
