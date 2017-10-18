# -*- coding:utf-8 -*- 
# Author: Roc-J

import argparse
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

# parse the input params
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compress the input image using clustering')
    parser.add_argument("--input-file", dest="input_file", required=True)
    parser.add_argument("--num-bits", dest="num_bits", required=False, type=int, help="Number of bits used to represent each pixel")
    return parser

def compress_image(img, num_clusters):
    '''
    将输入的图片转换成（样本量，特征量）数组，然后再使用k-means聚类算法
    :param img:
    :param num_clusters:
    :return:
    '''
    X = img.reshape((-1, 1))

    # 对输入数据进行k-means
    kmeans =  KMeans(n_clusters=num_clusters, n_init=4, random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    # 为每个数据配置离它最近的中心点，并转变为图片的形状
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)

    return input_image_compressed

# plot the image
def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()

    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits

    if not 1 <= num_bits <= 8:
        raise TypeError('Number of bits should be between 1 and 8')

    num_clusters = np.power(2, num_bits)

    compress_rate = round(100.0 * (8.0 - args.num_bits)/8.0, 2)
    print "\nThe size of the image will be reduce by a factor of", 8.0/args.num_bits
    print "\nCompression rate =", compress_rate, '%'

    # load the image
    input_image = misc.imread(input_file, True).astype(np.uint8)

    # show the origin picture
    plot_image(input_image, 'Original image')

    # compress
    input_image_compressed = compress_image(input_image, num_clusters)
    plot_image(input_image_compressed, 'Compressed image; compression rate = '+ str(compress_rate) + '%')
    plt.show()