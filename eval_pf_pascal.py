from __future__ import print_function, division
import os
from os.path import exists
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from lib.model import ImMatchNet
from lib.pf_dataset import PFPascalDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import BatchTensorToVars, str_to_bool
from lib.point_tnf import corr_to_matches
from lib.eval_util import pck_metric
from lib.dataloader import default_collate
from lib.torch_util import collate_custom
from lib import pf_pascal_dataset as pf
from lib import tools

import argparse
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("NCNet evaluation script - PF Pascal dataset")

    use_cuda = torch.cuda.is_available()

    # Argument parsing
    parser = argparse.ArgumentParser(description="Compute PF Pascal matches")
    parser.add_argument("--checkpoint", type=str, default="models/ancnet_86_11.pth.tar")
    parser.add_argument(
        "--vis",
        type=int,
        default=0,
        help="visilisation options: 0 calculate pck; 1 visualise keypoint matches and heat maps; 2 display matched key points",
    )
    parser.add_argument("--a", type=float, default=0.1, help="a is the pck@alpha value")
    parser.add_argument(
        "--num_examples", type=int, default=5, help="the number of matching examples"
    )

    args = parser.parse_args()

    vis = args.vis
    alpha = args.a
    num_examples = args.num_examples

    if args.checkpoint is not None and args.checkpoint is not "":
        print("Loading checkpoint...")
        checkpoint = torch.load(
            args.checkpoint, map_location=lambda storage, loc: storage
        )
        checkpoint["state_dict"] = OrderedDict(
            [
                (k.replace("vgg", "model"), v)
                for k, v in checkpoint["state_dict"].items()
            ]
        )

        args = checkpoint["args"]
    else:
        print("checkpoint needed.")
        exit()

    cnn_image_size = (args.image_size, args.image_size)

    # Create model
    print("Creating CNN model...")
    model = ImMatchNet(
        use_cuda=use_cuda,
        feature_extraction_cnn=args.backbone,
        checkpoint=checkpoint,
        ncons_kernel_sizes=args.ncons_kernel_sizes,
        ncons_channels=args.ncons_channels,
        pss=args.pss,
        noniso=args.noniso,
    )
    model.eval()

    print("args.dataset_image_path", args.dataset_image_path)
    # Dataset and dataloader
    collate_fn = default_collate
    csv_file = "image_pairs/test_pairs.csv"

    dataset = PFPascalDataset(
        csv_file=os.path.join(args.dataset_image_path, csv_file),
        dataset_path=args.dataset_image_path,
        transform=NormalizeImageDict(["source_image", "target_image"]),
        output_size=cnn_image_size,
    )
    dataset.pck_procedure = "scnet"

    # Only batch_size=1 is supported for evaluation
    batch_size = 1

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

    # initialize vector for storing results
    stats = {}
    stats["point_tnf"] = {}
    stats["point_tnf"]["pck"] = np.zeros((len(dataset), 1))

    # Compute pck accuracy
    total = len(dataloader)
    progress = tqdm(dataloader, total=total)
    for i, batch in enumerate(progress):
        batch = batch_tnf(batch)
        batch_start_idx = batch_size * i
        corr4d = model(batch)

        # get matches
        # note invert_matching_direction doesnt work at all
        xA, yA, xB, yB, sB = corr_to_matches(
            corr4d, do_softmax=True, invert_matching_direction=False
        )

        matches = (xA, yA, xB, yB)
        stats = pck_metric(
            batch, batch_start_idx, matches, stats, alpha=alpha, use_cuda=use_cuda
        )

    # Print results
    results = stats["point_tnf"]["pck"]
    good_idx = np.flatnonzero((results != -1) * ~np.isnan(results))
    print("Total: " + str(results.size))
    print("Valid: " + str(good_idx.size))
    filtered_results = results[good_idx]
    print("PCK:", "{:.2%}".format(np.mean(filtered_results)))

    test_csv = "test_pairs.csv"
    dataset_val = pf.ImagePairDataset(
        transform=NormalizeImageDict(["source_image", "target_image"]),
        dataset_image_path=args.dataset_image_path,
        dataset_csv_path=os.path.join(args.dataset_image_path, "image_pairs"),
        dataset_csv_file=test_csv,
        output_size=cnn_image_size,
        keypoints_on=True,
        original=True,
        test=True,
    )
    loader_test = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4)
    batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

    print("visualise correlation")
    tools.visualise_feature(
        model, loader_test, batch_tnf, image_size=cnn_image_size, MAX=num_examples
    )
    print("visualise pair")
    tools.validate(
        model,
        loader_test,
        batch_tnf,
        None,
        image_scale=args.image_size,
        im_fe_ratio=16,
        image_size=cnn_image_size,
        MAX=num_examples,
        display=True,
    )


if __name__ == "__main__":
    main()
