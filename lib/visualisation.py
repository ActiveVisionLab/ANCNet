from . import constant
import numpy as np
import matplotlib.pyplot as plt


def displaySingle(img0, key0, file_name):
    plt.imshow(img0)
    cm_hot = plt.get_cmap('hsv') #coolwarm
    plt.axis('off')

    N = key0.shape[0]
    # dispaly bounding boxes
    num=0
    for i in range(N):
        tmp = key0[i]
        if tmp[2]> 0.1:
            num+=1
            plt.scatter(tmp[0],tmp[1], s=100, 
                        edgecolors='w',
                        color=cm_hot(float(num/N)), 
                        alpha=1., 
                        marker='o')

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()

def displaySingle2(img0, key0, key_gt, file_name):
    plt.imshow(img0)
    cm_hot = plt.get_cmap('tab20') #coolwarm
    plt.axis('off')

    N = key0.shape[0]
    # dispaly bounding boxes
    num=0
    for i in range(N):
        tmp = key0[i]
        gt = key_gt[i]
        if tmp[2]> 0.1:
            num+=1
            plt.scatter(tmp[0],tmp[1], s=100, 
                        edgecolors='w',
                        color=cm_hot(float(num/N)), 
                        alpha=.5, 
                        marker='X')
            plt.scatter(gt[0],gt[1], s=100, 
                        edgecolors='w',
                        color=cm_hot(float(num/N)), 
                        alpha=.5, 
                        marker='o')
            x = [tmp[0],gt[0]]
            y = [tmp[1], gt[1]]
            plt.plot(x, y, alpha=.5, color=cm_hot(float(num/N)),linewidth=3.0)

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()


def displayPair(img0, k0, img1, k1, gt1, file_name=None):
    """
    displayPair() visualise a pair of image to be matched
    Arguments:
        img0, img1 [numpy.array float] H X W x 3: source and target image
        k0, k1 [numpy.array float] N x 3: source and target key points
                        the first and second column are x y coordinate in image
                        the third column is an indicator: for k0, the index of key 
                        points; for k1, the indicator can be messy and therefore to
                        be set according to its source. the k0 and k1 may contain
                        empty zero rows, but the rows for k0 and k1 must be synchronized
        is_axisoff: bool switcher for showing axis
    """
    tmp = np.zeros((k1.shape[0], 3), float)
    tmp[:, :2] = k1
    k1 = tmp
    for a0, a1 in zip(k0, k1):
        if a0[2] > 0:
            a1[2] = a0[2]
        else:
            a1[2] = 0.0

    # source image
    displaySingle(img0, k0, file_name+'source.png')
    displaySingle(img1, k1, file_name+'target.png')
    displaySingle2(img1, k1, gt1, file_name+'prediction.png')

