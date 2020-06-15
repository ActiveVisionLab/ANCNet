from PIL import Image
import torch.nn.functional as F
import os, time, sys, math
import subprocess, shutil
from . import constant
from . import visualisation
from os.path import *
import numpy as np
import numpy
import torch
import tqdm as tqdm
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from . import interpolator


def seed_torch(seed=1029):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir != "" and not exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, join(model_dir, str(state["epoch"]) + "_" + model_fn))


def calc_gt_indices(batch_keys_gt, batch_assignments_gt):
    """
        calc_gt_indices() calculate the ground truth indices and number of valid key points.
        Arguments:
            batch_keys_gt: [tensor B x N x 3] the last column stores the indicator of the key points
                            if it is larger than 0 it is valid, otherwise invalid
            batch_assignments_gt [tensor B x N x N], the ground truth assignment matrix
        Returns:
            indices_gt: [tensor B x N ]: from source to target, each element stores the index of target matches
            key_num_gt: [tensor B]: number of valid key points per input of the batch
    """
    _, indices_gt = torch.max(
        batch_assignments_gt, 2
    )  # get ground truth matches from source to target
    indices_gt += (
        1
    )  # remember that indices start counting from 1 for 0 is used to store empty key points
    mask_gt = (batch_keys_gt[:, :, 2] > 0).long()  # get the valid key point masks
    indices_gt = indices_gt * mask_gt
    key_num_gt = mask_gt.sum(dim=1).float()
    return indices_gt, key_num_gt


def calc_accuracy(batch_assignments, indices_gt, src_key_num_gt):
    """
    calc_accuracy() calculate the accuracy for each instance in a batch of ground truth key points
                    and batch and predicted assignments.
    Arguments:
        batch_assignment [tensor float B x 32 x 32]: the batch of the predicted assignment matrix
        indices_gt [tensor long B x 32 ]: the batch of ground truth indices from source to target
        src_key_num_gt [tensor float Bx 1]: the ground truth number of valid key points for a batch
                                            with batch size B.
    Returns:
        accuracy [tensor float B x 32]: the accuracy for each instance of the batch is calculated.
    """

    values, indices = torch.max(
        batch_assignments, 2
    )  # get matches for source key points
    indices += (
        1
    )  # remember that indices start counting from 1 for 0 is used to store empty key points

    accuracy = (indices_gt == indices).sum(dim=1).float()
    accuracy = torch.div(accuracy, src_key_num_gt)

    return accuracy


def pure_pck(keys_pred, keys_gt, key_num_gt, image_scale, alpha):
    """
    pure_pck() calculate the pck percentage for each instance in a batch of predicted key points
    Arguments:
        keys_pred [tensor float B x 32 x 3]: the predicted key points
        keys_gt [tensor float B x 32 x 2]: the ground truth key points. 
        key_num_gt [tensor float B X 1]: the ground truth number of valid key points
        image_scale float: the length of image diagonal image_scale = sqrt( W^2 + H^2)
        alpha float: percentage threshold.
    Returns:
        pck [tensor float Bx 1]: the pck score for the batch
    """
    dif = keys_pred - keys_gt
    err = dif.norm(dim=2) / image_scale
    wrong = (err > alpha).sum(dim=1).float()  # number of incorrect predictior
    pck = 1 - torch.div(wrong, key_num_gt)
    return pck


def calc_pck0(
    target_batch_keys,
    target_keys_pred,
    batch_assignments_gt,
    src_key_num_gt,
    image_scale,
    alpha=0.1,
):
    """
    calc_pck() calculate the pck percentage for each instance in a batch of predicted key points
    Arguments:
        target_batch_keys [tensor float B x 32 x 3]: the target key points
        target_keys_pred [tensor float B x 32 x 2]: the predicted key points. 
        batch_assignments_gt [tensor float B x 32 x 32]: the ground truth assignment matrix.
        src_key_num_gt [tensor float B X 1]: the ground truth number of valid key points
        image_scale float: the length of image diagonal image_scale = sqrt( W^2 + H^2)
        alpha float: percentage threshold.
    Returns:
        pck [tensor float Bx 1]: the pck score for the batch
        target_keys_pred, [tensor float B x 2], predicted key points locations w.r.t. target image
        batch_keys_gt, [tensor float B x 2 ], ground truth key points locations w.r.t. target image
    """
    batch_keys_gt = torch.bmm(batch_assignments_gt, target_batch_keys[:, :, :2])
    pck = pure_pck(target_keys_pred, batch_keys_gt, src_key_num_gt, image_scale, alpha)
    return pck, target_keys_pred, batch_keys_gt


def distance(keys_pred, keys_gt, key_num_gt):
    """
    pure_pck() calculate the pck percentage for each instance in a batch of predicted key points
    Arguments:
        keys_pred [tensor float B x 32 x 3]: the predicted key points
        keys_gt [tensor float B x 32 x 2]: the ground truth key points. 
        key_num_gt [tensor float B X 1]: the ground truth number of valid key points
    Returns:
        pck [tensor float Bx 1]: the pck score for the batch
    """
    mask = (keys_gt > 1e-10).float()
    dif = keys_pred * mask - keys_gt
    err = dif.norm(dim=2)
    err = err.sum(dim=1)
    err = torch.div(err, key_num_gt)
    return err


def calc_distance(
    target_batch_keys, target_keys_pred, batch_assignments_gt, src_key_num_gt
):
    """
    calc_pck() calculate the pck percentage for each instance in a batch of predicted key points
    Arguments:
        target_batch_keys [tensor float B x 32 x 3]: the target key points
        target_keys_pred [tensor float B x 32 x 2]: the predicted key points. 
        batch_assignments_gt [tensor float B x 32 x 32]: the ground truth assignment matrix.
        src_key_num_gt [tensor float B X 1]: the ground truth number of valid key points
        image_scale float: the length of image diagonal image_scale = sqrt( W^2 + H^2)
        alpha float: percentage threshold.
    Returns:
        pck [tensor float B x 1]: the pck score for the batch
        target_keys_pred, [tensor float B x 2], predicted key points locations w.r.t. target image
        batch_keys_gt, [tensor float B x 2 ], ground truth key points locations w.r.t. target image
    """
    batch_keys_gt = torch.bmm(batch_assignments_gt, target_batch_keys[:, :, :2])
    err = distance(target_keys_pred, batch_keys_gt, src_key_num_gt)
    return err


def calc_pck(
    target_batch_keys,
    batch_assignments,
    batch_assignments_gt,
    src_key_num_gt,
    image_scale,
    alpha=0.1,
):
    """
    calc_pck() calculate the pck percentage for each instance in a batch of predicted key points
    Arguments:
        target_batch_keys [tensor float B x 32 x 3]: the target key points
        batch_assignments [tensor float B x 32 x 32]: the predicted assignment matrix. 
        batch_assignments_gt [tensor float B x 32 x 32]: the ground truth assignment matrix.
        src_key_num_gt [tensor float B X 1]: the ground truth number of valid key points
        image_scale float: the length of image diagonal image_scale = sqrt( W^2 + H^2)
        alpha float: percentage threshold.
    Returns:
        pck [tensor float Bx 1]: the pck score for the batch
        batch_keys_pred, [tensor float B x 2], predicted key points locations w.r.t. target image
        batch_keys_gt, [tensor float B x 2 ], ground truth key points locations w.r.t. target image
    """

    batch_keys_pred = torch.bmm(batch_assignments, target_batch_keys[:, :, :2])
    return calc_pck0(
        target_batch_keys,
        batch_keys_pred,
        batch_assignments_gt,
        src_key_num_gt,
        image_scale,
        alpha,
    )


def calc_mto(batch_assignments, src_indices_gt, src_key_num_gt):
    """
    calc_mto() calculate the one-to-many matching score, notice one is source, many is destination
    Arguments:
        batch_assignment [tensor float B x 32 x 32]: the batch of the predicted assignment matrix
        src_indices_gt [tensor long B x 32 ]: the batch of ground truth key point indices from source to target
        src_key_num_gt [tensor float Bx 1]: the ground truth number of valid key points for a batch with batch size B
    Returns:
        mto [tensor B x 1] cpu: the mto score for the batch    
    """
    values, indices = torch.max(batch_assignments, 2)
    indices += (
        1
    )  # remember that indices start counting from 1 for 0 is used to store empty key points
    mask = (src_indices_gt == indices).long()
    indices *= mask
    num_unique = torch.tensor([float(len(torch.unique(kk))) - 1 for kk in indices])
    mto = 1 - torch.div(num_unique, src_key_num_gt.cpu())
    return mto


def graph_matching(batch_assignments, iterations=2):
    """
    graph_matching() applying the graph matching update to refine the matches
    Arguments:
        batch_assignment [tensor float B x 32 x 32]: the batch of the predicted assignment matrix
    Returns:
        batch_assignment [tensor float B x 32 x 32]: the batch of the refined assignment matrix
    """

    for i in range(iterations):
        batch_assignments = batch_assignments * batch_assignments
        Xrs_sum = torch.sum(batch_assignments, dim=2, keepdim=True)  # normalisation
        batch_assignments = batch_assignments / (Xrs_sum + constant._eps)
    return batch_assignments


def corr_to_matches(corr4d, do_softmax=False, source_to_target=True):
    B, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    XA, YA = np.meshgrid(range(fs2), range(fs1))  # pixel coordinate
    XB, YB = np.meshgrid(range(fs4), range(fs3))
    XA, YA = torch.FloatTensor(XA), torch.FloatTensor(YA)
    XB, YB = torch.FloatTensor(XB), torch.FloatTensor(YB)
    XA, YA = XA.view(-1).cuda(), YA.view(-1).cuda()
    XB, YB = XB.view(-1).cuda(), YB.view(-1).cuda()

    if source_to_target:
        # best match from source to target
        nc_A_Bvec = corr4d.view(B, fs1, fs2, fs3 * fs4)

        if do_softmax:
            nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec, dim=3)

        match_A_vals, idx_A_Bvec = torch.max(nc_A_Bvec, dim=3)  # B x fs1 x fs2
        score = match_A_vals.view(B, 1, fs1, fs2)  # B x 1 x fs1 x fs2

        # idx_A_Bvec: 12 x 16 x 16
        xB = XB[idx_A_Bvec.view(-1)].view(
            B, 1, fs1, fs2
        )  # B x fs1*fs2: index of betch matches in B
        yB = YB[idx_A_Bvec.view(-1)].view(B, 1, fs1, fs2)
        xyB = torch.cat((xB, yB), 1)

        return xyB.contiguous(), score.contiguous()
    else:
        # best matches from target to source
        nc_B_Avec = corr4d.view(B, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
        if do_softmax:  # default
            nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, dim=1)

        match_B_vals, idx_B_Avec = torch.max(
            nc_B_Avec, dim=1
        )  # idx_B_Avec is Bx (16*16 = 256)
        score = match_B_vals.view(B, 1, fs3, fs4)  # score is B x 1 x fs3 x fs4

        # idx_B_Avec: 12 x 16 x 16
        xA = XA[idx_B_Avec.view(-1)].view(
            B, 1, fs3, fs4
        )  # B x 256, it stores the col index for A
        yA = YA[idx_B_Avec.view(-1)].view(
            B, 1, fs3, fs4
        )  # B x 256, it stores the col index for A
        xyA = torch.cat((xA, yA), 1)

        return xyA.contiguous(), score.contiguous()


def NormalisationPerRow(keycorr):
    """
    NormalisationPerRow() normalise the 3rd dimension by calculating its sum and divide the vector 
            in last dimension by the sum
    Arguments
        keycorr: B x N x HW 
    Returns 
        keycorr: B x N x HW
    """
    eps = 1e-15
    sum_per_row = keycorr.sum(dim=2, keepdim=True) + eps
    sum_per_row = sum_per_row.expand_as(keycorr)  # B x N x L
    keycorr = keycorr / sum_per_row
    return keycorr


class ExtractFeatureMap:
    def __init__(self, im_fe_ratio):
        self.im_fe_ratio = im_fe_ratio
        self.interp = interpolator.Interpolator(im_fe_ratio)
        self.offset = int(im_fe_ratio / 2 - 1)

    def upsampling_keycorr(self, keycorr, image_size):
        B, N, H, W = keycorr.shape
        keycorr = F.interpolate(
            keycorr, size=image_size, mode="bilinear", align_corners=False
        )  # (H2-1)*interp.im_fe_ratio+2) x (W2-1)*interp.im_fe_ratio+2)
        keycorr = keycorr.view(B, N, -1).contiguous()
        return keycorr

    def normalise_image(self, keycorr_original, kmin=None, krange=None):
        keycorr = keycorr_original.clone()
        eps = 1e-15
        B, N, C = keycorr.shape
        keycorr = keycorr.view(B, -1)
        if kmin is None and krange is None:
            kmin, _ = keycorr.min(dim=1, keepdim=True)  # B x 1
            kmax, _ = keycorr.max(dim=1, keepdim=True)
            krange = kmax - kmin
            krange = krange.expand_as(keycorr)
            kmin = kmin.expand_as(keycorr)
        keycorr = (keycorr - kmin) / krange
        return keycorr.view(B, N, C), kmin, krange

    def __call__(self, corr, key_gt, source_to_target=True, image_size=None):
        """
        extract_featuremap() extract the interpolated feature map for each query key points in key_gt
        Arguements    
            corr [tensor float] B x 1 x H1 x W1 x H2 x W2: the 4d correlation map
            key_gt [tensor float] B x N x 2: the tensor stores the sparse query key points 
            image_size [tuple int] H, W: original input image size, if it is None, then no interpolation  
            interp [object of Interpolator]: to interpolate the correlation maps
            source_to_targe [boolean]: if true, query from source to target, otherwise, from target to source 
        Return:
            keycorr [tensor float]: B x N x H2W2 (when source_to_targe = True) the correlation map for each source
                                    key ponit 
        """
        B, C, H1, W1, H2, W2 = corr.shape
        _, N, _ = key_gt.shape
        if source_to_target:
            corr = corr.view(B, H1, W1, H2 * W2)
            corr = corr.permute(0, 3, 1, 2)
            keycorr = self.interp(corr, key_gt)  # keycorr B x H2*W2 x N, key is source
            keycorr = keycorr.permute(0, 2, 1)  # B x N x H2*W2
            keycorr = keycorr.view(B, N, H2, W2)  # B x N x H2 x W2
        else:
            corr = corr.view(B, H1 * W1, H2, W2)
            keycorr = self.interp(
                corr, key_gt
            )  # keycorr B x H1*W1 x N key_gt is target point
            keycorr = keycorr.permute(0, 2, 1)  # B x N x H1*W1
            keycorr = keycorr.view(B, N, H1, W1)  # B x N x H1 x W1

        if image_size is not None:
            keycorr = self.upsampling_keycorr(keycorr, image_size)
        keycorr = keycorr.view(B, N, -1).contiguous()
        # try softmax
        # keycorr = torch.softmax(keycorr, dim = 2)
        keycorr = NormalisationPerRow(keycorr)
        return keycorr

    def keycorr_to_matches(self, keycorr, image_size):
        """
            keycorr_to_matches() 
            keycorr [tensor float]: B x N x HW (when source_to_targe = True) the correlation map for each source
                                    key ponit, note H x W must be aligned with the original image size
            image_size (tuple) H x W: original image size                                    
            Returns 
                xyA [tensor float]: B x N x 2, the key points from source to target                                 
        """
        B, N, _ = keycorr.shape

        XA, YA = np.meshgrid(
            range(image_size[1]), range(image_size[0])
        )  # pixel coordinate
        XA, YA = (
            torch.FloatTensor(XA).view(-1).cuda(),
            torch.FloatTensor(YA).view(-1).cuda(),
        )

        values, indices = torch.max(keycorr, dim=2)
        xA = XA[indices.view(-1)].view(B, N, 1)
        yA = YA[indices.view(-1)].view(B, N, 1)
        xyA = torch.cat((xA, yA), 2)
        return xyA


def validate(
    model,
    loader,
    batch_preprocessing_fn,
    graph_layer,
    image_scale,
    im_fe_ratio=16,
    image_size=(256, 256),
    alpha=0.1,
    MAX=100,
    display=False,
    iterations=2,
):
    model.train(mode=False)
    avg_recall = 0.0

    total = min(MAX, len(loader))

    mean_accuracy = 0
    mean_accuracy2 = 0
    mean_accuracy3 = 0
    mean_mto = 0
    mean_mto2 = 0
    mean_mto3 = 0
    mean_pck = 0
    mean_pck2 = 0
    mean_pck3 = torch.zeros((1, 1))
    mean_pck4 = torch.zeros((1, 1))

    output_dir = "output"
    if output_dir != "" and not exists(output_dir):
        os.makedirs(output_dir)

    extract_featuremap = ExtractFeatureMap(im_fe_ratio)
    progress = tqdm(loader, total=total)
    for i, data in enumerate(progress):
        if i >= total:
            break
        tnf_batch = batch_preprocessing_fn(data)
        corr = model(tnf_batch)  # Xr is the predicted permutation matrix

        Xg = tnf_batch["assignment"]
        Xgt = tnf_batch["assignment"].permute(0, 2, 1)

        src_gt = tnf_batch["source_points"]
        dst_gt = tnf_batch["target_points"]
        # calc key_num
        src_indices_gt, src_key_num_gt = calc_gt_indices(src_gt, Xg)
        dst_indices_gt, dst_key_num_gt = calc_gt_indices(dst_gt, Xgt)

        keycorrB_A = extract_featuremap(
            corr, src_gt[:, :, :2], source_to_target=True, image_size=image_size
        )
        keycorrA_B = extract_featuremap(
            corr, dst_gt[:, :, :2], source_to_target=False, image_size=image_size
        )
        xyB_A = extract_featuremap.keycorr_to_matches(keycorrB_A, image_size)
        xyA_B = extract_featuremap.keycorr_to_matches(keycorrA_B, image_size)

        pck3, src_key_p3, src_key_gt3 = calc_pck0(
            dst_gt, xyB_A, Xg, dst_key_num_gt, image_scale, alpha
        )
        pck4, src_key_p4, src_key_gt4 = calc_pck0(
            src_gt, xyA_B, Xgt, src_key_num_gt, image_scale, alpha
        )
        mean_pck3 += pck3.mean()
        mean_pck4 += pck4.mean()

        # visualise results
        if display:

            source = (tnf_batch["source_original"] * 255).int()
            target = (tnf_batch["target_original"] * 255).int()
            B = source.shape[0]
            for b in range(B):
                file_name = join(output_dir, "{}_{}_".format(i, b))
                visualisation.displayPair(
                    source[b].detach().cpu().permute(1, 2, 0),
                    src_gt[b].detach().cpu(),
                    target[b].detach().cpu().permute(1, 2, 0),
                    src_key_p3[b].detach().cpu(),
                    src_key_gt3[b].detach().cpu(),
                    file_name=file_name,
                )

    mean_pck3 /= total
    mean_pck4 /= total
    model.train(mode=True)
    return mean_pck3, mean_pck4


def visualise_feature(
    model, loader, batch_preprocessing_fn, image_size, im_fe_ratio=16, MAX=100
):
    model.train(mode=False)

    total = min(MAX, len(loader))

    extract_featuremap = ExtractFeatureMap(im_fe_ratio)
    progress = tqdm(loader, total=total)
    cm_hot = plt.get_cmap("bwr")  # coolwarm

    output_dir = "output"
    if output_dir != "" and not exists(output_dir):
        os.makedirs(output_dir)

    for i, data in enumerate(progress):
        if i >= total:
            break
        tnf_batch = batch_preprocessing_fn(data)

        category = tnf_batch["set"]
        Xg = tnf_batch["assignment"]
        src_gt_cuda = tnf_batch["source_points"]
        dst_gt_cuda = tnf_batch["target_points"]
        src_indices_gt, src_key_num_gt = calc_gt_indices(src_gt_cuda, Xg)

        src_gt = src_gt_cuda.detach().cpu()
        dst_gt = dst_gt_cuda.detach().cpu()
        target = (tnf_batch["target_original"] * 255).int()
        source = (tnf_batch["source_original"] * 255).int()
        B, N, _ = src_gt.shape

        corr = model(tnf_batch)  # Xr is the predicted permutation matrix
        B, C, H1, W1, H2, W2 = corr.shape

        keycorrB_A = extract_featuremap(
            corr, src_gt_cuda[:, :, :2], source_to_target=True, image_size=image_size
        )
        keycorrA_B = extract_featuremap(
            corr, dst_gt_cuda[:, :, :2], source_to_target=False, image_size=image_size
        )
        xyB_A = extract_featuremap.keycorr_to_matches(keycorrB_A, image_size)
        xyA_B = extract_featuremap.keycorr_to_matches(keycorrA_B, image_size)
        keycorrB_A, _, _ = extract_featuremap.normalise_image(keycorrB_A)
        keycorrA_B, _, _ = extract_featuremap.normalise_image(keycorrA_B)

        keycorrB_A = keycorrB_A.view(B, N, *image_size).detach().cpu()
        keycorrA_B = keycorrA_B.view(B, N, *image_size).detach().cpu()
        xyB_A = xyB_A.detach().cpu()
        xyA_B = xyA_B.detach().cpu()

        for b in range(B):
            NN = min(32, int(src_key_num_gt[b]))
            fig, axes = plt.subplots(4, NN, sharex="all", sharey="all")
            nn = 0
            for n in range(N):

                tn = src_indices_gt[b, n]  # target index
                if tn > 0:
                    # paint source
                    c = n % len(constant._colors)
                    m = n // len(constant._colors)

                    original = source[b].detach().cpu().permute(1, 2, 0)

                    axes[0, nn].imshow(original)
                    axes[0, nn].scatter(
                        src_gt[b, n, 0],
                        src_gt[b, n, 1],
                        s=5.,
                        edgecolors='g',
                        color='g',
                        alpha=.7,
                        marker='o',
                    )
                    axes[0, nn].axis("off")

                    # paint 4d corr
                    im = cm_hot(keycorrB_A[b, n].detach().cpu()) * 255
                    original = target[b].detach().cpu().permute(1, 2, 0).numpy()
                    im = im[:, :, :3] * 0.5 + original * 0.5
                    im = np.uint8(im)
                    im = Image.fromarray(im)

                    axes[1, nn].imshow(im)
                    axes[1, nn].scatter(
                        xyB_A[b, n, 0],
                        xyB_A[b, n, 1],
                        s=1.,
                        edgecolors="r",
                        color='r',
                        alpha=.7,
                        marker='o',
                    )

                    axes[1, nn].scatter(
                        dst_gt[b, tn - 1, 0],
                        dst_gt[b, tn - 1, 1],
                        s=1.,
                        edgecolors='g',
                        color='g',
                        alpha=.7,
                        marker='o',
                    )
                    axes[1, nn].axis("off")

                    # paint target
                    axes[2, nn].imshow(target[b].detach().cpu().permute(1, 2, 0))
                    axes[2, nn].scatter(
                        dst_gt[b, tn - 1, 0],
                        dst_gt[b, tn - 1, 1],
                        s=5.,
                        edgecolors='g',
                        color='g',
                        alpha=.7,
                        marker='o',
                    )
                    axes[2, nn].axis("off")

                    im = cm_hot(keycorrA_B[b, tn - 1].detach().cpu()) * 255
                    original = source[b].detach().cpu().permute(1, 2, 0).numpy()
                    im = im[:, :, :3] * 0.5 + original * 0.5
                    im = np.uint8(im)
                    im = Image.fromarray(im)

                    axes[3, nn].imshow(im)
                    axes[3, nn].scatter(
                        xyA_B[b, tn - 1, 0],
                        xyA_B[b, tn - 1, 1],
                        s=1.,
                        edgecolors="r",
                        color='r',
                        alpha=.7,
                        marker='o',
                    )
                    axes[3, nn].scatter(
                        src_gt[b, n, 0],
                        src_gt[b, n, 1],
                        s=1.,
                        edgecolors='g',
                        color='g',
                        alpha=.7,
                        marker='o',
                    )
                    axes[3, nn].axis("off")

                    nn += 1
                    if nn >= NN:
                        break

            # source image
            file_name = join(output_dir, "{}_heatmaps.png".format(i))
            plt.axis("off")
            plt.savefig(
                file_name, bbox_inches="tight", pad_inches=0, quality=100, dpi=1200
            )
            plt.clf()
            # plt.show()
