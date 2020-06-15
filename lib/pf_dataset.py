from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from lib.transformation import AffineTnf
import matplotlib.pyplot as plt


class PFPascalDataset(Dataset):

    """
    
    Proposal Flow PASCAL image pair dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(
        self,
        csv_file,
        dataset_path,
        output_size=(240, 240),
        transform=None,
        category=None,
        pck_procedure="pf",
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
        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.category = self.pairs.iloc[:, 2].values.astype("float")
        if category is not None:
            cat_idx = np.nonzero(self.category == category)[0]
            self.category = self.category[cat_idx]
            self.pairs = self.pairs.iloc[cat_idx, :]
        self.img_A_names = self.pairs.iloc[:, 0]
        self.img_B_names = self.pairs.iloc[:, 1]
        self.point_A_coords = self.pairs.iloc[:, 3:5]
        self.point_B_coords = self.pairs.iloc[:, 5:]
        self.dataset_path = dataset_path
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = AffineTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)
        self.pck_procedure = pck_procedure

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A, im_size_A = self.get_image(self.img_A_names, idx)
        image_B, im_size_B = self.get_image(self.img_B_names, idx)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)

        # compute PCK reference length L_pck (equal to max bounding box side in image_A)
        # L_pck = torch.FloatTensor([torch.max(point_A_coords.max(1)[0]-point_A_coords.min(1)[0])])
        N_pts = torch.sum(torch.ne(point_A_coords[0, :], -1))

        if self.pck_procedure == "pf":
            L_pck = torch.FloatTensor(
                [
                    torch.max(
                        point_A_coords[:, :N_pts].max(1)[0]
                        - point_A_coords[:, :N_pts].min(1)[0]
                    )
                ]
            )
        elif self.pck_procedure == "scnet":
            # modification to follow the evaluation procedure of SCNet
            point_A_coords[0, 0:N_pts] = (
                point_A_coords[0, 0:N_pts] * self.out_w / im_size_A[1]
            )
            point_A_coords[1, 0:N_pts] = (
                point_A_coords[1, 0:N_pts] * self.out_h / im_size_A[0]
            )

            point_B_coords[0, 0:N_pts] = (
                point_B_coords[0, 0:N_pts] * self.out_w / im_size_B[1]
            )
            point_B_coords[1, 0:N_pts] = (
                point_B_coords[1, 0:N_pts] * self.out_h / im_size_B[0]
            )

            im_size_A[0:2] = torch.FloatTensor([self.out_h, self.out_w])
            im_size_B[0:2] = torch.FloatTensor([self.out_h, self.out_w])

            L_pck = torch.FloatTensor([self.out_h])

        sample = {
            "source_image": image_A,
            "target_image": image_B,
            "source_im_size": im_size_A,
            "target_im_size": im_size_B,
            "source_points": point_A_coords,
            "target_points": point_B_coords,
            "L_pck": L_pck,
        }

        # # get key points annotation
        # np_img_A = sample['source_image'].long().numpy().transpose(1,2,0)
        # np_img_B = sample['target_image'].long().numpy().transpose(1,2,0)

        # kp_A = sample['source_points'].transpose(1,0).numpy()
        # kp_B = sample['target_points'].transpose(1,0).numpy()

        # kp_A[:,0] *= self.out_w/float(im_size_A[1])
        # kp_A[:,1] *= self.out_h/float(im_size_A[0])
        # print('kp_A', kp_A)
        # print('L_pck', sample['L_pck'])
        # fig=plt.figure(figsize=(1, 2))
        # ax0 = fig.add_subplot(1, 2, 1)
        # # ax0.add_patch(rect)
        # plt.imshow(np_img_A)
        # # dispaly bounding boxes
        # for i, kp in enumerate(kp_A):
        #     if kp[0] == kp[0]:
        #         ax0.scatter(kp[0],kp[1], s=5, color='r',alpha=1.)
        # ax1 = fig.add_subplot(1, 2, 2)
        # # rect = matplotlib.patches.Rectangle((bbox_B[0],bbox_B[1]),bbox_B[2]-bbox_B[0],bbox_B[3]-bbox_B[1],linewidth=1,edgecolor='r',facecolor='none')
        # # ax1.add_patch(rect)
        # plt.imshow(np_img_B)
        # for i, kp in enumerate(kp_B):
        #     if kp[0] == kp[0]:
        #         ax1.scatter(kp[0],kp[1], s=5, color='r',alpha=1.)
        # plt.show()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.dataset_path, img_name_list.iloc[idx])
        image = io.imread(img_name)

        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        im_size = torch.Tensor(im_size.astype(np.float32))

        return (image, im_size)

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20)), axis=0
        )

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))

        return point_coords

