import tensorflow as tf

path = '/home/elahe/NortfaceProject/codes/datasets/cifar10/train/Airplane/aeroplane_s_000004.png'
sample = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
sample = tf.keras.preprocessing.image.img_to_array(sample)

print(sample.shape)

