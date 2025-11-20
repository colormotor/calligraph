#!/usr/bin/env python3
"""
)   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

Stroke initialization utils

"""

from . import geom, segmentation, imaging
from easydict import EasyDict as edict
import numpy as np
import pdb
from PIL import ImageOps


def softmax(x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum() 


def init_path_tsp(img, n, nb_iter=30,
                  saliency_type='intensity',
                  mask=None,
                  startup_w=None,
                  closed=False,
                  minutes_limit=1/30, **kwargs):
    from . import tsp_art, segmentation
    # Saliency
    if type(img)==np.ndarray:
        img = Image.fromarray((img*255).astype(np.uint8))
    if saliency_type=='ood':
        sal = segmentation.ood_saliency(img.convert('RGB'))[0]
    elif saliency_type=='clip':
        sal = segmentation.clip_saliency(input_img)
    else:
        sal = np.array(img.convert('L'))/255
        
    density_map = ((sal-sal.min())/(sal.max() - sal.min()))**2 #(sal*(1-img))
    if mask is not None:
        density_map *= mask
    #density_map = density_map*(1-img)
    points = tsp_art.weighted_voronoi_sampling(density_map, n, nb_iter=nb_iter)
    # Find top left point
    n = len(points)
    P = [tuple(p) for p in points]
    # sorted according to x,y
    I = sorted(list(range(n)), key=lambda i: P[i])
    points = np.array([points[i] for i in I])
    # TSP func assumes first and last points are fixed if cylce is True and end_to_end is True
    I = tsp_art.heuristic_solve(points,
                                time_limit_minutes=minutes_limit,
                                cycle=closed,
                                end_to_end=not closed,
                                **kwargs) #, logging=True, verbose=True)
    P = points[I]
    if startup_w is not None:
        P = np.hstack([P, np.ones((len(P), 1))*startup_w])
    return [P], density_map


def init_paths_random(img, n, num_ctrl, sigma=2.5, k=7, tau=0.2, use_attention=True, intersect_xdog=True, width=None, get_data=False, sample_around_radius=0):
    if use_attention:
        attn_map = segmentation.clip_saliency(img) #painter.input_img)
        edges = 1-segmentation.xdog(img, sigma=sigma, k=k)
    else:
        edges = 1
        attn_map = np.array(img)
        if np.max(attn_map) > 1:
            attn_map = attn_map/255
        attn_map = 1-attn_map
        edges = np.array(attn_map)

    if intersect_xdog:
        attn_map *= edges
    attn_map_soft = np.copy(attn_map)
    attn_map_soft[attn_map > 0] = softmax(attn_map[attn_map > 0], tau=tau)
    paths = []

    def attn_val(p):
        x, y = p
        return attn_map[int(y), int(x)]
    def attn_dist(a, b):
        v = attn_val((a + b)/2)
        return (1-v) + geom.distance(a, b)/img.width

    if sample_around_radius > 0:
        inds = np.random.choice(range(attn_map.flatten().shape[0]), size=n, replace=False, p=attn_map_soft.flatten())
        inds = np.array(np.unravel_index(inds, attn_map.shape))[::-1].T
        P = inds.astype(np.float64)

        for p in P:
            blob = []
            for t in np.linspace(0, np.pi*2, num_ctrl, endpoint=False):
                blob.append(p + np.array([np.cos(t), np.sin(t)])*np.random.uniform(0.2, 1.0)*sample_around_radius)
            blob = np.array(blob)
            if width is not None:
                blob = np.hstack([blob, np.ones((len(blob), 1))*width])
            paths.append(blob)
    else:
        for i in range(n):
            inds = np.random.choice(range(attn_map.flatten().shape[0]), size=num_ctrl, replace=False, p=attn_map_soft.flatten())
            inds = np.array(np.unravel_index(inds, attn_map.shape))[::-1].T
            P = inds.astype(np.float64)
            if width is not None:
                P = np.hstack([P, np.ones((len(P), 1))*width])
            paths.append(P)

    if not get_data:
        return paths
    return edict(dict(paths=paths,
                      saliency=attn_map,
                      edges=edges))
