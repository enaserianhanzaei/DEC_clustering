import numpy as np
from DEC import DEC
import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import os
import shutil
from clustering import kmeans, arrange_clustering
from sklearn.cluster import KMeans


def onlytransfer(n_clusters, fb_kmeans=True):

    X = np.load('data/chan/8chan_pol/VGG16/fc1/featuresx.npy')
    X = X.astype('float32')
    pathfile = open('data/chan/8chan_pol/VGG16/fc1/paths.txt', "r")
    pathlist = pathfile.readlines()
    pathlist = [path[:-1] for path in pathlist]
    pathfile.close()

    if fb_kmeans:
        #features = torch.from_numpy(features)
        images_lists, loss = kmeans(X, nmb_clusters=n_clusters, preprocess=False)
        Y_pred = arrange_clustering(images_lists)
    else:
        km = KMeans(n_clusters=n_clusters, n_init=20)
        Y_pred = km.fit_predict(X)

    for y_pred, path in zip(Y_pred, pathlist):
        savedir = '/home/elahe/NortfaceProject/codes/DEC-keras/results/clusters/8chan_pol/%s/%s/%s' % ('transfer', 'fc1', y_pred)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        shutil.copy(path, savedir)



def apply_final_model(n_clusters=15):
    #X = np.load('/home/elahe/NortfaceProject/codes/DEC-keras/data/8chan_pol/original/HOG/featuresx.npy')
    #pathfile = open('/home/elahe/NortfaceProject/codes/DEC-keras/data/8chan_pol/original/HOG/paths.txt', "r")
    X = np.load('./data/chan/8chan_pol/VGG16/fc1/featuresx.npy')
    pathfile = open('./data/chan/8chan_pol/VGG16/fc1/paths.txt', "r")

    pathlist = pathfile.readlines()
    pathlist = [path[:-1] for path in pathlist]
    pathfile.close()

    print(len(pathlist), X.shape)

    init = 'glorot_uniform'
    #weights = './results/models/8chan_pol/HOG/DEC_model_final_%s.h5' % n_clusters
    weights = './results/models/chan/8chan_pol/fc1/DEC_model_final_%s.h5' % n_clusters
    dec = DEC(dims=[X.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
    dec.model.load_weights(weights)
    count = 0

    q = dec.model.predict(X, verbose=0)
    Y_pred = q.argmax(1)

    print(np.unique(Y_pred))
    for y_pred, path in zip(Y_pred, pathlist):
        #x = np.reshape(x, newshape=(1, 200704))#8916, #4096
        #q = dec.model.predict(x, verbose=0)
        #y_pred = q.argmax(1)[0]

        savedir = '/home/elahe/NortfaceProject/codes/DEC-keras/results/clusters/8chan_pol/%s/%s/%s' % ('VGG', 'fc1', y_pred)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        count += 1

        shutil.copy(path, savedir)

    print(count)


def main():
    x = np.load('./data/chan/all/VGG/featuresx.npy')

    init = 'glorot_uniform'

    # prepare the DEC model
    silhouette_avgs = []
    nims = []
    rel_loss = []
    prev = None
    for n_clusters in [5, 10, 15, 20]:
        weights = './results/models/chan/all/VGG/DEC_model_final_%s.h5' % n_clusters
        dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
        dec.model.load_weights(weights)

        q = dec.model.predict(x, verbose=0)
        y_pred = q.argmax(1)
        if not prev is None:
            nmi_ = np.round(metrics.nmi(prev, y_pred), 5)
            nims.append((nmi_))
            print('\n |==> NMI against previous assignment: {0:.3f} <==|'.format(nmi_))
        prev = y_pred

        silhouette_avg = silhouette_score(x, y_pred)
        silhouette_avgs.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        '''tr_loss = dec.model.evaluate(x)
        ts_loss = dec.model.evaluate(x_test)
        rel_loss.append(tr_loss/ts_loss)
        print('\n |==> relative loss: {0:.4f} <==|'.format(tr_loss/ts_loss))'''

    plt.plot(range(len(nims)), nims)
    plt.show()

    plt.plot(range(len(silhouette_avgs)), silhouette_avgs)
    plt.show()


if __name__ == '__main__':
    apply_final_model(n_clusters=15)
    #onlytransfer(n_clusters=15)