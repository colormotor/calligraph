#!/usr/bin/env python3

from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageFilter
from easydict import EasyDict as edict
import torch
import os

from calligraph import (plut,
                        geom,
                        svg,
                        config,
                        clipper,
                        imaging,
                        util,
                        fs,
                        diffvg_utils,
                        spline_losses,
                        image_losses,)

import pdb

import torch
device = config.device
dtype = torch.float32

def params():
    save_every = 10
    # NB output the layout will be created in the same directory as the image
    # E.g. elphant.jpg will become elphant_layout.json
    image_path = './data/silhouettes/SEAGULL.jpg' 
    image_path = './data/silhouettes/elephant.jpg' 
    output_path = './generated'

    # By default assume that the input filename contains the string
    text = os.path.splitext(os.path.basename(image_path))[0] # 
    font = 'UnYetgul'
    
    alpha =  0.5 
    image_alpha = 0.7 
    seed = 100
    size = 350

    subd = 100 #25 #8
    lr_pos = (size)/subd 
    lr_scale = 1.0/subd
    lr_rot = np.pi/subd

    ds = 2
    offset = 5
    sil_offset = 10 #10

    scale_range = (1.2, 2.5) #(1.2, 1.5)
    rot_range = geom.radians(20) #np.pi/6 # 0.1 # np.pi/7

    num_opt_steps = 200 #200 #300 #150 # 300 #000 # 500
    
    if alpha < 1:
        overlap_w = 1000 
    else:
        overlap_w = 0

    mse_w = 20 
    mse_mul = 2 # Factor multiplying each mse blur level (> 1 emph low freq)

    stroke_w = 2.0
    width_mul = 1.0
    scale = 1.0
    return locals()

cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = '___'
saver = util.SaveHelper(__file__, output_path,
                        cfg=cfg)


input_img = Image.open(cfg.image_path).convert('L')
w, h = input_img.size
box = geom.make_rect(0, 0, w, h)

im = np.array(input_img)/255
ctrs = clipper.multi_union([imaging.find_contours(1-im)]*2)
if cfg.sil_offset:
    area = clipper.offset(clipper.offset(ctrs, -cfg.sil_offset), cfg.sil_offset, join_type='round')
else:
    area = ctrs
rast = imaging.ShapeRasterizer(box, w)
rast.fill_shape(area)
input_img = rast.Image(invert=True)
img = np.array(input_img)/255

##############################################
# Settings
verbose = False
util.seed_everything(cfg.seed)

##############################################
# Target tensor
target_img = 1-(1.0-img)*(cfg.alpha*cfg.image_alpha)
target_tensor = torch.tensor(target_img, device=device, dtype=dtype).contiguous()
background_image = np.ones_like(target_img)

##############################################
# Create glyphs and create target glyph image
box = geom.bounding_box(area)

##############################################
# Initialization paths
from calligraph import ttf

db = ttf.FontDatabase(os.path.expanduser('~/Dropbox/fontdata/fonts'))
#print(db.font_names())

glyph_box = geom.scale_rect(box, 0.8) #make_rect(0, 0, w, h)
glyphs = db.shapes(cfg.font, cfg.text, spacing=1.2, merge=False)
S = sum(glyphs, [])
mat = geom.rect_in_rect_transform(geom.bounding_box(S), glyph_box)
glyphs = [geom.tsm(mat, S) for S in glyphs]
glyphs = [geom.fix_shape_winding(S) for S in glyphs]
glyphs_sampled = [[geom.uniform_sample(P, cfg.ds, closed=True) for P in S] for S in glyphs]

rast = imaging.ShapeRasterizer(box, w)
for glyph in glyphs:
    rast.fill_shape(glyph)
text_img = rast.Image(invert=True)

##############################################
# Create scene
scene = diffvg_utils.Scene()

ind = 0

glyphs_centered = []

for i, glyph in enumerate(glyphs_sampled):
    glyph = clipper.offset(glyph, cfg.offset, join_type='round')
    box = geom.bounding_box(glyph)
    p = geom.rect_center(box)
    glyph = geom.tsm(geom.trans_2d(-p), glyph)
    glyphs_centered.append(geom.tsm(geom.trans_2d(-p), glyphs[i]))
    shapes = []
    tsm = diffvg_utils.Transform((p, True), ([0.0], True), ([1.0], True))

    for X in glyph:
        shape = diffvg_utils.Path(points=X, degree=1, closed=True) 
        shapes.append(shape)
    scene.add_shapes(shapes, False, transform=tsm,
                     stroke_color=None, fill_color=([cfg.alpha], False))

##############################################
# Optimizer setup

params = [(scene.get_transform_positions(), cfg.lr_pos),
          (scene.get_transform_rotations(), cfg.lr_rot),
          (scene.get_transform_scales(), cfg.lr_scale),]

opt = diffvg_utils.SceneOptimizer(scene,
                                  params=params,
                                  num_steps=cfg.num_opt_steps,
                                  schedule_start=0.4,
                                  lr_min_scale=0.2)


##############################################
# Losses

mse = image_losses.MultiscaleMSELoss(rgb=False, debug=False)
opt.add_loss('mse',
                 image_losses.MultiscaleMSELoss(rgb=False), cfg.mse_w,
                 inputs=('im', 'input_img', 'mse_mul'))

def curvature(points):
    closed= False #cfg.closed
    points = points[:,:2]
    length = len(points)

    if closed:
        indices = torch.arange(length)
        indices_next = torch.roll(indices, -1)
    else:
        indices = torch.arange(length-1)
        indices_next = indices + 1 
    edges = torch.stack([indices, indices_next])
    points_ = points[edges]
    edges = points_[1] - points_[0]
    if closed:
        a = edges
        b = torch.roll(edges, -1, 0)
    else:
        a = edges[:-1]
        b = edges[1:]
    res = torch.arctan2(a[:,0]*b[:,1] - a[:,1]*b[:,0], a[:,0]*b[:,0] + a[:,1]*b[:,1])/(np.pi) 
    return torch.abs(res)

def pos_bending_loss(positions):
    points = torch.vstack(positions)
    return torch.sum(curvature(points))/len(positions)

opt.add_loss('bending',
           pos_bending_loss, 30.0, ('transform_positions',))

overlap_loss = spline_losses.make_overlap_loss(cfg.alpha, blur=0, subtract_widths=False)
opt.add_loss('overlap', overlap_loss, cfg.overlap_w, ('im',)) 

opt.add_loss('bbox', spline_losses.make_bbox_loss(geom.rect(0, 0, w, h)), 1.0,
            inputs=('points',))


##############################################
# Begin visualization and optimize

plut.set_theme()

fig = plt.figure(figsize=(8,6))
gs = GridSpec(2, 2, height_ratios=[1.0, 0.25])

first_frame = None
start_time = None
render_shapes = []

def frame(step):
    global first_frame, render_shapes, start_time

    opt.zero_grad()


    # Rasterize
    with util.perf_timer('render', verbose=verbose):
        im = opt.render(background_image)[:,:,0].to(device)

    opt.step(im=im,
             input_img=target_tensor,
             mse_mul=cfg.mse_mul,
             transform_positions=scene.get_transform_positions(),
             points=[prim.points for prim in scene.primitives])
    
    
    # Constrain
    with torch.no_grad():
        for tsm in scene.transforms:
            tsm.scale.data.clamp_(*cfg.scale_range) #0.8, 1.5)
            tsm.rot.data.clamp_(-cfg.rot_range, cfg.rot_range)

    if step != 0 and cfg.headless and (step%cfg.save_every != cfg.save_every-1):
        return

    # Viz
    im = im.detach().cpu().numpy()
    if first_frame is None:
        first_frame = im

    plt.subplot(gs[0,0])
    plt.title('Startup - seed:%d'%cfg.seed)
    amt = 1.0
    plt.imshow(first_frame*amt + target_img*(1-amt), cmap='gray', vmin=0, vmax=1)
    plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)

    plt.subplot(gs[0,1])
    plt.title('step %d'%(step))
    amt = 0.5
    plt.imshow(im * amt + target_img*(1-amt), cmap='gray', vmin=0, vmax=1)
    plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)

    plt.subplot(gs[1,:])
    plt.title('Loss')
    opt.plot(50)
    
    transforms = [tsm.shape_to_canvas().detach().cpu().numpy() for tsm in scene.transforms]

    if cfg.save:
        if step%cfg.save_every == cfg.save_every-1 or cfg.num_opt_steps==1:
            if saver.valid:
                scene.save_json(saver.with_ext('.json'), transforms=transforms, glyphs=glyphs_centered, text=cfg.text)
                plut.figure_image(adjust=False).save(saver.with_ext('.png'))
                saver.copy_file()
            layout_file = os.path.splitext(cfg.image_path)[0] + '_layout.json'
            fs.save_json({'glyphs':glyphs_centered, 'transforms':transforms, 'text':cfg.text}, layout_file)


plut.show_animation(fig, frame, cfg.num_opt_steps, filename=saver.with_ext('.mp4'), headless=cfg.headless)
