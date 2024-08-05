#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
from scene.cameras import Camera
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torch.fft as fft

def render_set(model_path, name, iteration, views: list, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    diff_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diff")
    fdiff_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fdiff")
    magnify_path = os.path.join(model_path, name, "ours_{}".format(iteration), "magnify")
    compen_path = os.path.join(model_path, name, "ours_{}".format(iteration), "compen")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(diff_path, exist_ok=True)
    makedirs(fdiff_path, exist_ok=True)
    makedirs(magnify_path, exist_ok=True)
    makedirs(compen_path, exist_ok=True)


    f_size = 4
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        img_h, img_w = rendering.shape[1],  rendering.shape[2]

        gt = view.original_image[0:3, :, :]

        diff: torch.Tensor = gt-rendering
        diff_mono = diff
        im_fft = fft.fft2(diff_mono.cpu())
        # print(im_fft.shape)
        # exit()
        im_shifted = fft.fftshift(im_fft)
        zero_freq = torch.zeros_like(im_shifted, dtype=torch.cfloat)
        zero_freq[:, int(img_h/2-f_size):int(img_h/2+f_size+1), int(img_w/2-f_size):int(img_w/2+f_size+1)] = im_shifted[:, int(img_h/2-f_size):int(img_h/2+f_size+1), int(img_w/2-f_size):int(img_w/2+f_size+1)]
        im_fft2 = fft.ifftshift(zero_freq)
        fdiff = fft.ifft2(im_fft2).real
        compensated = gt-fdiff.cuda()
        torchvision.utils.save_image(compensated, os.path.join(compen_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(gt+2*fdiff.cuda(), os.path.join(magnify_path, '{0:05d}'.format(idx) + ".png"))

        # diff_min = diff.min()
        diff_min = -0.69
        diff = diff-diff_min
        diff_max = diff.max()
        diff_max = 1.4
        # print(diff_max)
        diff = diff/diff_max
        fdiff = fdiff.cuda()-diff_min
        fdiff = fdiff/diff_max

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(diff, os.path.join(diff_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fdiff, os.path.join(fdiff_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)