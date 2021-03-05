import numpy as np
import faiss


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def kmeans(data, nmb_clusters, preprocess=True, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """

    if preprocess:
        x = preprocess_features(data)
    else:
        x = data

    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)


    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 20000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    # elahe https://github.com/facebookresearch/faiss/issues/1179
    # losses = faiss.vector_to_array(clus.obj)
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])

    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    I = [int(n[0]) for n in I]
    images_lists = [[] for i in range(nmb_clusters)]
    for i in range(len(data)):
        images_lists[I[i]].append(i)

    return images_lists, losses[-1]


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]