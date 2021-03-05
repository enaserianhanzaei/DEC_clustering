import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def dispImg(X, n, fname=None):
    h = X.shape[1]
    w = X.shape[2]
    c = X.shape[3]

    print(h, w, c, n)
    buff = np.zeros((n * w, n * w, c), dtype=np.uint8)#

    for i in range(n):
        for j in range(n):
            buff[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = X[i * n + j]

    if fname is None:
        cv2.imshow('a', buff)
        cv2.waitKey(0)
    else:
        cv2.imwrite(fname, buff)


def hog_picture(hog, resolution):
    import scipy.ndimage as imrotate
    glyph1 = np.zeros((resolution, resolution), dtype=np.uint8)
    glyph1[:, round(resolution / 2) - 1:round(resolution / 2) + 1] = 255
    glyph = np.zeros((resolution, resolution, 9), dtype=np.uint8)
    glyph[:, :, 0] = glyph1
    for i in range(1, 9):
        glyph[:, :, i] = imrotate.rotate(glyph1, -i * 20)

    shape = hog.shape
    clamped_hog = hog.copy()
    clamped_hog[hog < 0] = 0
    image = np.zeros((resolution * shape[0], resolution * shape[1]), dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(9):
                image[i * resolution:(i + 1) * resolution, j * resolution:(j + 1) * resolution] = np.maximum(
                    image[i * resolution:(i + 1) * resolution, j * resolution:(j + 1) * resolution],
                    clamped_hog[i, j, k] * glyph[:, :, k])

    return image


def load_stl(fname):
    from joblib import Parallel, delayed
    from PIL import Image
    from temp import features


    '''X = np.load(fname)
    #X = X.astype(np.uint8)
    print(X.shape)
    X = X.reshape((X.shape[0], 224, 224, 3))#2376
    print(X.shape)'''

    '''plt.imshow(X[0])
    plt.show()

    plt.imshow(X[0, :, :, 2], interpolation='nearest')
    plt.show()

    img = Image.fromarray(X[0])
    img.show()'''

    X = np.fromfile('/home/elahe/NortfaceProject/codes/DEC-keras/stl/' + fname, dtype=np.uint8)
    X = X.reshape((int(X.size / 3 / 96 / 96), 3, 96, 96)).transpose((0, 3, 2, 1))
    print(X.shape)
    dispImg(X[:100, :, :, [2, 1, 0]], 10, fname + '_org.jpg')


    n_jobs = 10
    cmap_size = (8, 8)
    N = X.shape[0]

    H = np.asarray(Parallel(n_jobs=n_jobs)(
        delayed(features.hog)(Image.fromarray(X[i])) for i in range(N)))
    #H = np.asarray(Parallel(n_jobs=n_jobs)(delayed(features.hog)(X[i]) for i in range(N)))

    #H_img = np.repeat(np.asarray([hog_picture(H[i], 9) for i in range(100)])[:, :, :, np.newaxis], 3, 3)
    #dispImg(H_img, 10, fname + '_hog.jpg')
    H = H.reshape((H.shape[0], int(H.size / N)))

    X_small = np.asarray(
        Parallel(n_jobs=n_jobs)(delayed(cv2.resize)(X[i], cmap_size) for i in range(N)))
    crcb = np.asarray(Parallel(n_jobs=n_jobs)(delayed(cv2.cvtColor)(X_small[i], cv2.COLOR_RGB2YCrCb) for i in range(N)))
    crcb = crcb[:, :, :, 1:]
    crcb = crcb.reshape((crcb.shape[0], int(crcb.size / N)))

    feature = np.concatenate(((H - 0.2) * 10.0, (crcb - 128.0) / 10.0), axis=1)
    print(feature.shape)

    return feature, X[:, :, :, [2, 1, 0]]


def make_stl_data():
  np.random.seed(1234)
  random.seed(1234)
  X_train, img_train = load_stl('train_X.bin')
  X_test, img_test = load_stl('test_X.bin')
  X_unlabel, img_unlabel = load_stl('unlabeled_X.bin')
  Y_train = np.fromfile('/home/elahe/NortfaceProject/codes/DEC-keras/stl/train_y.bin', dtype=np.uint8) - 1
  Y_test = np.fromfile('/home/elahe/NortfaceProject/codes/DEC-keras/stl/test_y.bin', dtype=np.uint8) - 1

  X_total = np.concatenate((X_train, X_test), axis=0)
  img_total = np.concatenate((img_train, img_test), axis=0)
  Y_total = np.concatenate((Y_train, Y_test))
  p = np.random.permutation(X_total.shape[0])
  X_total = X_total[p]
  img_total = img_total[p]
  Y_total = Y_total[p]
  #write_db(X_total, Y_total, 'stl_total')
  #write_db(img_total, Y_total, 'stl_img')

  X = np.concatenate((X_total, X_unlabel), axis=0)
  p = np.random.permutation(X.shape[0])
  X = X[p]
  Y = np.zeros((X.shape[0],))
  N = X.shape[0]*4/5

  #write_db(X[:N], Y[:N], 'stl_train')
  #write_db(X[N:], Y[N:], 'stl_test')


def load_chan(fname):
    from joblib import Parallel, delayed
    from PIL import Image
    from temp import features

    X = np.load('/home/elahe/NortfaceProject/codes/DEC-keras/data/8chan_pol/original/' + fname)
    print(X.shape)
    X = X.reshape((X.shape[0], 224, 224, 3))#2376
    print(X.shape)

    dispImg((X[:100, :, :, [2, 1, 0]] * 255).astype(np.uint8), 10, fname + '_org.jpg')

    n_jobs = 10
    cmap_size = (8, 8)
    N = X.shape[0]
    H = np.asarray(Parallel(n_jobs=n_jobs)(delayed(features.hog)(Image.fromarray((X[i] * 255).astype(np.uint8))) for i in range(N)))
    H = H.reshape((H.shape[0], int(H.size / N)))

    X_small = np.asarray(Parallel(n_jobs=n_jobs)(delayed(cv2.resize)((X[i] * 255).astype(np.uint8), cmap_size) for i in range(N)))
    crcb = np.asarray(Parallel(n_jobs=n_jobs)(delayed(cv2.cvtColor)(X_small[i], cv2.COLOR_RGB2YCrCb) for i in range(N)))
    crcb = crcb[:, :, :, 1:]
    crcb = crcb.reshape((crcb.shape[0], int(crcb.size / N)))

    feature = np.concatenate(((H - 0.2) * 10.0, (crcb - 128.0) / 10.0), axis=1)
    print(feature.shape)

    return feature, X[:, :, :, [2, 1, 0]]



def make_chan_data():
  np.random.seed(1234)
  random.seed(1234)
  X_train, img_train = load_chan('featuresx.npy')
  pathfile = open('/data/chan/8chan_pol/original/paths.txt', "r")
  pathlist = pathfile.readlines()
  pathlist = [path[:-1] for path in pathlist]
  pathfile.close()

  plt.imshow(img_train[100])
  plt.show()

  from PIL import Image
  im = Image.open(pathlist[100])
  im.show()

  np.save('/home/elahe/NortfaceProject/codes/DEC-keras/data/8chan_pol/original/HOG' + '/featuresx', X_train)
  with open('/home/elahe/NortfaceProject/codes/DEC-keras/data/8chan_pol/original/HOG' + '/paths.txt', 'w') as f:
      for item in pathlist:
          f.write("%s\n" % item)


def load_cifar(fname):
    from joblib import Parallel, delayed
    from PIL import Image
    from temp import features

    X = np.load('/home/elahe/NortfaceProject/codes/DEC-keras/data/cifar10/original/' + fname)
    print(X.shape)
    X = X.reshape((X.shape[0], 224, 224, 3))#2376
    print(X.shape)

    dispImg((X[:100, :, :, [2, 1, 0]] * 255).astype(np.uint8), 10, fname + '_org.jpg')

    n_jobs = 10
    cmap_size = (8, 8)
    N = X.shape[0]
    H = np.asarray(Parallel(n_jobs=n_jobs)(delayed(features.hog)(Image.fromarray((X[i] * 255).astype(np.uint8))) for i in range(N)))
    H = H.reshape((H.shape[0], int(H.size / N)))

    X_small = np.asarray(Parallel(n_jobs=n_jobs)(delayed(cv2.resize)((X[i] * 255).astype(np.uint8), cmap_size) for i in range(N)))
    crcb = np.asarray(Parallel(n_jobs=n_jobs)(delayed(cv2.cvtColor)(X_small[i], cv2.COLOR_RGB2YCrCb) for i in range(N)))
    crcb = crcb[:, :, :, 1:]
    crcb = crcb.reshape((crcb.shape[0], int(crcb.size / N)))

    feature = np.concatenate(((H - 0.2) * 10.0, (crcb - 128.0) / 10.0), axis=1)
    print(feature.shape)

    return feature, X[:, :, :, [2, 1, 0]]


def make_cifar_data():
  np.random.seed(1234)
  random.seed(1234)
  X_train, img_train = load_cifar('featuresx.npy')
  pathfile = open('/data/chan/8chan_pol/original/paths.txt', "r")
  pathlist = pathfile.readlines()
  pathlist = [path[:-1] for path in pathlist]
  pathfile.close()

  plt.imshow(img_train[100])
  plt.show()

  from PIL import Image
  im = Image.open(pathlist[100])
  im.show()

  np.save('/home/elahe/NortfaceProject/codes/DEC-keras/data/8chan_pol/original/HOG' + '/featuresx', X_train)
  with open('/home/elahe/NortfaceProject/codes/DEC-keras/data/8chan_pol/original/HOG' + '/paths.txt', 'w') as f:
      for item in pathlist:
          f.write("%s\n" % item)



