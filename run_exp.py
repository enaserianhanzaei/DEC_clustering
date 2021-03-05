#https://github.com/XifengGuo/DEC-keras
#https://github.com/piiswrong/dec/blob/e551e01f90a3d67d2ef9c90e968c8faf5d8f857d/dec/dec.py#L504
from DEC import DEC
import os, csv
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
import numpy as np

expdir='./results/pretrained/'
if not os.path.exists(expdir):
    os.mkdir(expdir)

logfile = open(expdir + '/results.csv', 'a')
logwriter = csv.DictWriter(logfile, fieldnames=['trials', 'acc', 'nmi', 'ari'])
logwriter.writeheader()

trials=1
for db in ['mnist']: #'usps', 'stl', 'reuters10k',, 'mnist', 'fmnist'
    logwriter.writerow(dict(trials=db, acc='', nmi='', ari=''))
    save_db_dir = os.path.join(expdir, db)
    if not os.path.exists(save_db_dir):
        os.mkdir(save_db_dir)

        # load dataset
    from datasets import load_data

    x, y, x_test, y_test = load_data(db)

    '''x, y = np.load('/home/elahe/NortfaceProject/codes/TransferLearning/keras/data/cifar10_featuresx_train.npy'), \
           np.load('/home/elahe/NortfaceProject/codes/TransferLearning/keras/data/cifar10_featuresy_train.npy')

    x_test, y_test = np.load(
        '/home/elahe/NortfaceProject/codes/TransferLearning/keras/data/cifar10_featuresx_test.npy'), \
                     np.load('/home/elahe/NortfaceProject/codes/TransferLearning/keras/data/cifar10_featuresy_test.npy')'''

    n_clusters = len(np.unique(y))

    print('number of clusters: ', n_clusters)

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    # setting parameters
    if db == 'mnist' or db == 'fmnist':
        update_interval = 30# 140
        pretrain_epochs = 10#300
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
    elif db == 'reuters10k':
        update_interval = 30
        pretrain_epochs = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
    elif db == 'usps':
        update_interval = 30
        pretrain_epochs = 50
    elif db == 'stl':
        update_interval = 30
        pretrain_epochs = 10
    elif db == 'cifar':
        update_interval = 30
        pretrain_epochs = 10

    # prepare model
    dims = [x.shape[-1], 500, 500, 2000, 10]

    '''Training for base and nosp'''
    results = np.zeros(shape=(trials, 3))
    baseline = np.zeros(shape=(trials, 3))
    metrics0=[]
    metrics1=[]
    for i in range(trials):  # base
        save_dir = os.path.join(save_db_dir, 'trial%d' % i)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
        dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer, nclusters=10,
                     epochs=pretrain_epochs,
                     save_dir=save_dir)
        dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        dec.fit(x, t_x=x_test, y=y, t_y=y_test, maxiter=200,
                update_interval=update_interval,
                save_dir=save_dir)

        log = open(os.path.join(save_dir, 'dec_log.csv'), 'r')
        reader = csv.DictReader(log)
        metrics = []
        for row in reader:
            metrics.append([row['acc'], row['nmi'], row['ari']])
        metrics0.append(metrics[0])
        metrics1.append(metrics[-1])
        log.close()

    metrics0, metrics1 = np.asarray(metrics0, dtype=float), np.asarray(metrics1, dtype=float)
    for t, line in enumerate(metrics0):
        logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], ari=line[2]))
    logwriter.writerow(dict(trials=' ', acc=np.mean(metrics0, 0)[0], nmi=np.mean(metrics0, 0)[1], ari=np.mean(metrics0, 0)[2]))
    for t, line in enumerate(metrics1):
        logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], ari=line[2]))
    logwriter.writerow(dict(trials=' ', acc=np.mean(metrics1, 0)[0], nmi=np.mean(metrics1, 0)[1], ari=np.mean(metrics1, 0)[2]))

logfile.close()