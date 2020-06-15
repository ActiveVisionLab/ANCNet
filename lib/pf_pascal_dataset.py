from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
import numpy as np
from . import transformation as tf
import scipy.io
import matplotlib
import matplotlib.pyplot as plt


class ImagePairDataset(Dataset):

    """
    
    Image pair dataset used for weak supervision
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        training_image_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(
        self,
        dataset_csv_path,
        dataset_csv_file,
        dataset_image_path,
        dataset_size=0,
        output_size=(240, 240),
        transform=None,
        random_crop=False,
        keypoints_on=False,
        original=True,
        test=False,
    ):
        self.category_names = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.random_crop = random_crop
        self.out_h, self.out_w = output_size
        self.annotations = os.path.join(
            dataset_image_path, "PF-dataset-PASCAL", "Annotations"
        )
        self.train_data = pd.read_csv(os.path.join(dataset_csv_path, dataset_csv_file))
        if dataset_size is not None and dataset_size != 0:
            dataset_size = min((dataset_size, len(self.train_data)))
            self.train_data = self.train_data.iloc[0:dataset_size, :]
        self.img_A_names = self.train_data.iloc[:, 0]
        self.img_B_names = self.train_data.iloc[:, 1]
        self.set = self.train_data.iloc[:, 2].values
        self.test = test
        if self.test == False:
            self.flip = self.train_data.iloc[:, 3].values.astype("int")
        self.dataset_image_path = dataset_image_path
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = tf.AffineTnf(
            out_h=self.out_h, out_w=self.out_w, use_cuda=False
        )  # resize
        self.keypoints_on = keypoints_on
        self.original = original

    def __len__(self):
        return len(self.img_A_names)

    def __getitem__(self, idx):
        # get pre-processed images
        image_set = self.set[idx]
        if self.test == False:
            flip = self.flip[idx]
        else:
            flip = False

        cat = self.category_names[image_set - 1]

        image_A, im_size_A, kp_A, bbox_A = self.get_image(
            self.img_A_names, idx, flip, category_name=cat
        )
        image_B, im_size_B, kp_B, bbox_B = self.get_image(
            self.img_B_names, idx, flip, category_name=cat
        )
        A, kp_A = self.get_gt_assignment(kp_A, kp_B)

        sample = {
            "source_image": image_A,
            "target_image": image_B,
            "source_im_size": im_size_A,
            "target_im_size": im_size_B,
            "set": image_set,
            "source_points": kp_A,
            "target_points": kp_B,
            "source_bbox": bbox_A,
            "target_bbox": bbox_B,
            "assignment": A,
        }

        if self.transform:
            sample = self.transform(sample)

        if self.original:
            sample["source_original"] = image_A
            sample["target_original"] = image_B
        # # get key points annotation
        # np_img_A = sample['source_original'].numpy().transpose(1,2,0)
        # np_img_B = sample['target_original'].numpy().transpose(1,2,0)
        # print('bbox_A', bbox_A)
        # print('bbox_B', bbox_B)
        # rect = matplotlib.patches.Rectangle((bbox_A[0],bbox_A[1]),bbox_A[2]-bbox_A[0],bbox_A[3]-bbox_A[1],linewidth=1,edgecolor='r',facecolor='none')
        # print(rect)

        # fig=plt.figure(figsize=(1, 2))
        # ax0 = fig.add_subplot(1, 2, 1)
        # ax0.add_patch(rect)
        # plt.imshow(np_img_A)
        # # dispaly bounding boxes
        # for i, kp in enumerate(kp_A):
        #     if kp[0] == kp[0]:
        #         ax0.scatter(kp[0],kp[1], s=5, color='r',alpha=1.)
        # ax1 = fig.add_subplot(1, 2, 2)
        # rect = matplotlib.patches.Rectangle((bbox_B[0],bbox_B[1]),bbox_B[2]-bbox_B[0],bbox_B[3]-bbox_B[1],linewidth=1,edgecolor='r',facecolor='none')
        # print(rect)
        # ax1.add_patch(rect)
        # plt.imshow(np_img_B)
        # for i, kp in enumerate(kp_B):
        #     if kp[0] == kp[0]:
        #         ax1.scatter(kp[0],kp[1], s=5, color='r',alpha=1.)
        # plt.show()

        return sample

    def get_gt_assignment(self, kp_A, kp_B):
        """
            get_gt_assigment() get the ground truth assignment matrix
            Arguments:
                kp_A [Tensor, float32] Nx3: ground truth key points from the source image
                kp_B [Tensor, float32] Nx3: ground truth key points from the target image
            Returns:
                A [Tensor, float32] NxN: ground truth assignment matrix  
                kp_A [Tensor, float32] Nx3: ground truth key points + change original idx into target column idx
        """
        s = kp_A[:, 2].long()
        t = kp_B[:, 2].long()
        N = s.shape[0]
        A = torch.zeros(N, N)
        for n in range(N):
            if s[n] == 0:
                continue
            idx = (t == s[n]).nonzero()
            if idx.nelement() == 0:
                continue
            A[n, idx] = 1
            kp_A[n, 2] = idx + 1

        return A, kp_A

    def get_image(self, img_name_list, idx, flip, category_name=None):
        img_name = os.path.join(self.dataset_image_path, img_name_list.iloc[idx])
        image = io.imread(img_name)

        # if grayscale convert to 3-channel image
        if image.ndim == 2:
            image = np.repeat(np.expand_dims(image, 2), axis=2, repeats=3)

        if self.keypoints_on:
            keypoints, bbox = self.get_annotations(
                img_name_list.iloc[idx], category_name
            )

        # do random crop
        if self.random_crop:
            h, w, c = image.shape
            top = np.random.randint(h / 4)
            bottom = int(3 * h / 4 + np.random.randint(h / 4))
            left = np.random.randint(w / 4)
            right = int(3 * w / 4 + np.random.randint(w / 4))
            image = image[top:bottom, left:right, :]

        # get image size
        im_size = np.asarray(image.shape)

        # flip horizontally if needed
        if flip:
            image = np.flip(image, 1)
            if self.keypoints_on:
                N, _ = keypoints.shape
                for n in range(N):
                    if keypoints[n, 2] > 0:
                        keypoints[n, 0] = im_size[1] - keypoints[n, 0]
                bbox[0] = im_size[1] - bbox[0]
                bbox[2] = im_size[1] - bbox[2]
                tmp = bbox[0]
                bbox[0] = bbox[2]
                bbox[2] = tmp

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(
            0
        )  # the resized image becomes 400 x 400
        im_size = torch.Tensor(im_size.astype(np.float32))  # original image sise

        if self.keypoints_on:
            keypoints[:, 0] = keypoints[:, 0] / float(im_size[1]) * float(self.out_w)
            keypoints[:, 1] = keypoints[:, 1] / float(im_size[0]) * float(self.out_h)
            bbox[0] = bbox[0] / float(im_size[1]) * float(self.out_w)
            bbox[1] = bbox[1] / float(im_size[0]) * float(self.out_h)
            bbox[2] = bbox[2] / float(im_size[1]) * float(self.out_w)
            bbox[3] = bbox[3] / float(im_size[0]) * float(self.out_h)
            return (image, im_size, keypoints, bbox)
        else:
            return (image, im_size)

    def construct_graph(self, kp):
        """
        construct_graph() construct a sparse graph represented by G and H. 
        Arguments:
            kp [np array float, N x 3] stores the key points
        Returns
            G [np.array float, 32 x 96]: stores nodes by edges, if c-th edge leaves r-th node   
            H [np.array float, 32 x 96]: stores nodes by edges, if c-th edge ends at r-th node
        """
        N = kp.shape[0]

        G = np.zeros(32, 96)
        H = np.zeros(32, 96)
        return G, H

    def get_annotations(self, keypoint_annotation, category_name):
        """
            get_annotations() get key points annotation
            Arguments:
                keypoint_annotations str: the file name of the key point annotations
                category_name str: the category name of the image
            Returns:
                keypoint [Tensor float32] 32x3
                bbox [Tensor float32] 4
        """
        base, _ = os.path.splitext(os.path.basename(keypoint_annotation))
        # print('base', os.path.join(self.annotations, category_name, base +'.mat'))
        anno = scipy.io.loadmat(
            os.path.join(self.annotations, category_name, base + ".mat")
        )
        keypoint = np.zeros((32, 3), dtype=np.float32)
        annotation = anno["kps"]
        N = annotation.shape[0]
        for i in range(N):
            if (
                annotation[i, 0] == annotation[i, 0]
                and annotation[i, 1] == annotation[i, 1]
            ):  # not nan
                keypoint[i, :2] = annotation[i]
                keypoint[i, 2] = i + 1

        np.random.shuffle(keypoint)

        keypoint = torch.Tensor(keypoint.astype(np.float32))
        bbox = anno["bbox"][0].astype(np.float32)
        return keypoint, bbox


class ImagePairDatasetKeyPoint(ImagePairDataset):
    def __init__(
        self,
        dataset_csv_path,
        dataset_csv_file,
        dataset_image_path,
        dataset_size=0,
        output_size=(240, 240),
        transform=None,
        random_crop=False,
    ):
        super(ImagePairDatasetKeyPoint, self).__init__(
            dataset_csv_path,
            dataset_csv_file,
            dataset_image_path,
            dataset_size=dataset_size,
            output_size=output_size,
            transform=transform,
            random_crop=random_crop,
        )

