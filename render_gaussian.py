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

import math
import torch
import sys, os
from torch import nn
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_post
from scene import Scene, GaussianModel
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from lpipsPyTorch import lpips
from utils.graphics_utils import getProjectionMatrix, focal2fov
import numpy as np
import time
import cv2
import json
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights


def get_combined_args(parser : ArgumentParser, args=None):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline, _ = parser.parse_known_args(cmdlne_string)
    merged_dict = vars(args_cmdline).copy()
    if args is not None:
        for k,v in vars(args).items():
            if v != None:
                merged_dict[k] = v
    args_cmdline = Namespace(**merged_dict)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at", args_cmdline)
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)

def readTransform(path):
    transform={}
    with open(path, 'r') as file:
        data = json.load(file)
        transform["translation"]=np.array([data["x"],data["y"],data["z"]])
        qx, qy, qz, qw = data["qx"],data["qy"],data["qz"],data["qw"]
        transform["rotation"] = np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
        ])
        # correction = np.array([[0.9993908,  0.0000000,  0.0348995],
        #                     [0.0006091,  0.9998477, -0.0174418],
        #                     [-0.0348942,  0.0174524,  0.9992386]] )
        # transform["rotation"] = np.matmul(transform["rotation"], correction)
        transform["scale"]=data["scale"]
        return transform

class View(nn.Module):
    def __init__(self, R, T, focal_x, focal_y, width, height, data_device = "cuda" ):
        super(View, self).__init__()


        self.R = R
        self.T = T
        self.focal_y = focal_x
        self.focal_x = focal_y
        self.image_width = width
        self.image_height = height
        self.FoVx = focal2fov(focal_x,width)
        self.FoVy = focal2fov(focal_y,height)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01
        
        self.getWorldView(self.R,T)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, primx=0.5, primy=0.5).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3].cuda()
    
    def getWorldView(self, R, T):
        self.world_view_transform = np.zeros((4, 4))
        self.world_view_transform[3, 3] = 1.0
        t_inv = np.matmul(-R.transpose(),T)
        self.world_view_transform[:3,:3]=R.transpose()
        self.world_view_transform[:3,3] = t_inv
        self.world_view_transform = torch.tensor(np.float32(self.world_view_transform)).transpose(0, 1).cuda()
        
class Renderer():
    def __init__(self, args=None):
        # Set up command line argument parser
        parser = ArgumentParser(description="Testing script parameters")
        
        self.model = ModelParams(parser, sentinel=True)
        self.pipeline = PipelineParams(parser)
        parser.add_argument("--tau", default=0, type=int, help="Level of detail (0-15)")
        args = get_combined_args(parser, args)
        self.transform = readTransform(args.transform_GS)

        camera_model_path = args.camera_model
        with open(camera_model_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                data = line.split(" ")
                break
        self.camera_model = { "fl_x": float(data[4]),
        "fl_y": float(data[5]),
        "w": float(data[2]),
        "h": float(data[3])}

        dataset = self.model.extract(args)
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.gaussians.active_sh_degree = dataset.sh_degree
        self.scene = Scene(dataset, self.gaussians, resolution_scales = [1], create_from_hier=True)
        self.tau = args.tau
        self.train_test_exp=args.train_test_exp


    def render_view(self, rotation, translation):
        render_indices = torch.zeros(self.gaussians._xyz.size(0)).int().cuda()
        parent_indices = torch.zeros(self.gaussians._xyz.size(0)).int().cuda()
        nodes_for_render_indices = torch.zeros(self.gaussians._xyz.size(0)).int().cuda()
        interpolation_weights = torch.zeros(self.gaussians._xyz.size(0)).float().cuda()
        num_siblings = torch.zeros(self.gaussians._xyz.size(0)).int().cuda()

        #Apply transform to the camera
        rotation = self.transform['rotation']@rotation
        translation = (self.transform['scale'])*(self.transform['rotation']@translation)+self.transform['translation']


        viewpoint = View(rotation, translation, self.camera_model["fl_x"], self.camera_model["fl_y"], self.camera_model["w"], self.camera_model["h"])
        tanfovx = math.tan(viewpoint.FoVx * 0.5)
        threshold = (2 * (self.tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)
        to_render = expand_to_size(
            self.scene.gaussians.nodes,
            self.scene.gaussians.boxes,
            threshold,
            viewpoint.camera_center,
            torch.zeros((3)),
            render_indices,
            parent_indices,
            nodes_for_render_indices)
        
        indices = render_indices[:to_render].int().contiguous()
        node_indices = nodes_for_render_indices[:to_render].contiguous()
        get_interpolation_weights(
            node_indices,
            threshold,
            self.scene.gaussians.nodes,
            self.scene.gaussians.boxes,
            viewpoint.camera_center.cpu(),
            torch.zeros((3)),
            interpolation_weights,
            num_siblings
        )

        # Render the image
        image = torch.clamp(render_post(
            viewpoint, 
            self.scene.gaussians, 
            self.pipeline, 
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
            render_indices=indices,
            parent_indices = parent_indices,
            interpolation_weights = interpolation_weights,
            num_node_kids = num_siblings, 
            use_trained_exp=self.train_test_exp
            )["render"], 0.0, 1.0)

        image = image.data.cpu().numpy()
        image *= 255
        return image.astype(np.uint8)

if __name__ == "__main__":
    # Test render 
    renderer = Renderer()
    R = np.array([[-0.9439490687833039, -0.012658517779477973, 0.32984832494763394], [-0.32913601383730196, -0.039867014542148096, -0.9434405681052662], [0.025092627172631866, -0.9991248085595327, 0.03346605716920529]])
    T = np.array([-12.574966256566968, 38.94981550209616, -0.06905938699614564])
    start=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
    for _ in range(30):
        image=renderer.render_view(R, T)
    end=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
    print("Mean rendering time (ms): {} FPS: {}".format((end-start)*10**-6/30,(1/((end-start)*10**-9)*30)))
    image = image.swapaxes(0, 2).swapaxes(0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", image)
