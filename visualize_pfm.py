# -*- coding: utf-8 -*-
from sys import argv, path
from os import getcwd
from importlib import import_module
path.append(getcwd() + '/..')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# iphone SE 2nd gen
_CAMERA_PARAM = np.array([
    [796.0253508207987, 0,                 419.2138978530454],
    [0,                 796.9492651516134, 232.76788558751645],
    [0,                 0,                 1]])

_DIST_COEFF=np.array([
    0.3029725631848056,
    -2.097230226059333,
    -0.002075075172655089,
    0.003046368441015257,
    4.852914617528773])


def main(pfm_filepath: str, rgb_filepath: str, ax):
    m = import_module('utils')
    depth, _ = m.read_pfm(pfm_filepath)
    rgb = np.array(Image.open(rgb_filepath))

    new_k = cv2.getOptimalNewCameraMatrix(_CAMERA_PARAM, _DIST_COEFF, (depth.shape[1], depth.shape[0]), 1)[0]
    map_ = cv2.initUndistortRectifyMap(_CAMERA_PARAM, _DIST_COEFF, np.eye(3), new_k, (depth.shape[1], depth.shape[0]), cv2.CV_32FC1)

    fx = new_k[0, 0]
    fy = new_k[1, 1]
    cx = new_k[0, 2]
    cy = new_k[1, 2]

    depth = cv2.remap(depth, map_[0], map_[1], cv2.INTER_AREA)
    rgb = cv2.remap(rgb, map_[0], map_[1], cv2.INTER_AREA)

    point_cloud = None
    rgb_list = list()
    for v in range(depth.shape[0]):
        if v % 10 != 0:
            continue
        for u in range(depth.shape[1]):
            z = depth[v, u] / 1000.
            if u % 10 != 0:
                continue
            if z <= 0.0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            point = np.array([[x], [y], [z]])
            if point_cloud is None:
                point_cloud = point
            else:
                point_cloud = np.hstack([point_cloud, point])

            rgb_values = [rgb[v, u][0], rgb[v, u][1], rgb[v, u][2]]
            rgb_list.append(rgb_values)
    rgb_list = np.array(rgb_list)

    ax.set_xlim(np.min(point_cloud), np.max(point_cloud))
    ax.set_ylim(np.min(point_cloud), np.max(point_cloud))
    ax.set_zlim(np.min(point_cloud), np.max(point_cloud))
    obj = ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2], marker='.', c=rgb_list/255.)

    return obj


if __name__ == '__main__':
    pfm_filepath = argv[1]
    rgb_filepath = argv[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    main(pfm_filepath, rgb_filepath, ax)

    plt.show()
