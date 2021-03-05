import os
import extract_features
print('Building HOG feature extractor...')
#os.system('python setup_features.py build')
#os.system('python setup_features.py install')

print('Preparing stl data. This could take a while...')
extract_features.make_cifar_data()
