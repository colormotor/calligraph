#!/usr/bin/env python3
'''
DEMO:
Stroke-based image abstaction with score distillation sampling
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, ImageFilter
from easydict import EasyDict as edict
import torch, time, os

from calligraph import (
    plut,
    geom,
    bezier,
    dce,
    config,
    util,
    fs,
    diffvg_utils,
    imaging,
    stroke_init,
    segmentation,
    spline_losses,
    image_losses,
)

device = config.device
dtype = torch.float32


def params():
    # Script parameters
    # These are automatically converted to named script arguments
    # Currently requires booleans to be 0 or 1 to work from cmd-line

    verbose = 0
    output_path = './generated'

    text = 'G'
    size = 256
    padding = 15
    font = "./data/fonts/UnifrakturMaguntia-Regular.ttf"
    image_path = "./data/spock.jpg"

    minw, maxw = 0.0, 3.5  # stroke width range
    degree = 5
    deriv = 3
    multiplicity = 1
    b_spline = 1
    pspline = 0
    if deriv >= degree:
        pspline = 1
    catmull = 1
    ref_size_factor = 0.5
    fill = 1
    closed = 0
    if not closed:
        fill = 0
    seed = 133
    alpha =  1
    image_alpha = 0.5 # Determines density of coverage

    style_img = './data/style_imgs/zcal7.jpg'

    # For single paths
    point_density = 0.015 #0.003 # 0.015
    # For multiple paths
    num_paths = 30 #30
    num_vertices_per_path = 25
    spread_radius = 15

    lr_pos = 2.0
    lr_width = 0.3
    num_opt_steps = 300

    smoothing_w = 30
    use_clipag = 1
    style_w = 10
    distortion_scale = 0.3
    patch_size = 128
    blur = 0

    mse_w = 20.0
    mse_mul = 1

    sw = 2.0
    vary_width = 1

    # If true final video is the output only
    image_movie = True
    save_every = 10
    return locals()


# Parse command line and update parameters
cfg = util.ConfigArgs(params())

output_path = cfg.output_path
if not cfg.save:
    output_path = '___'

saver = util.SaveHelper(__file__, output_path,
                        cfg=cfg)

if cfg.text:
    input_img, outline = plut.font_to_image(cfg.text, image_size=(cfg.size, cfg.size), padding=cfg.padding,
                                            font_path=cfg.font, return_outline=True)
    img = np.array(input_img)/255
else:
    input_img = Image.open(cfg.image_path).convert('L').resize((cfg.size, cfg.size))
    img = np.array(input_img)/255
    
h, w = img.shape

style_img = Image.open(cfg.style_img).convert('L').resize((512, 512))
target_img = 1-(1.0-img)*(cfg.alpha*cfg.image_alpha)

##############################################
# Settings
background_image = np.ones((h, w))

##############################################
# Initialization paths
num_points = int(np.sum(1-img)*cfg.point_density)
startup_paths, _ = stroke_init.init_path_tsp(1-img, num_points, closed=cfg.closed, startup_w=cfg.sw)
    
print("Done")
# Add multiplicity
startup_paths = [np.kron(P, np.ones((cfg.multiplicity, 1))) for P in startup_paths]

##############################################
# Create the scene

scene = diffvg_utils.Scene()

fill_color = None
if cfg.fill:  #cfg.closed:
    fill_color=([0.0, 0.0, 0.0, cfg.alpha*cfg.image_alpha], False)

paths = []
for Pw in startup_paths:
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(Pw[:,:2],
                stroke_width=(Pw[:,2], True),
                degree=cfg.degree,
                pspline=cfg.pspline,
                split_pieces=cfg.alpha < 1 and not cfg.fill, #cfg.overlap_w > 0, #overlap,
                closed=cfg.closed)
    else:
        if cfg.catmull:
            path = diffvg_utils.CardinalSpline(Pw[:,:2],
                                               stroke_width=(Pw[:,2], True),
                                           closed=cfg.closed)
        else:
            Q = diffvg_utils.cardinal_spline(Pw, 0.5, closed=cfg.closed) #bezier.cubic_bspline_to_bezier_chain(Pw, periodic=cfg.closed)
            if cfg.closed:
                Q = Q[:-1]
            path = diffvg_utils.Path(Q[:,:2],
                                    stroke_width=(Q[:,2], True),
                                    degree=3,
                                    closed=cfg.closed)
    paths.append(path)
scene.add_shapes(paths, stroke_color=([cfg.alpha], False), fill_color=fill_color,
                     split_primitives=False )


##############################################
# Optimization

params = [(scene.get_points(), cfg.lr_pos)]
if cfg.vary_width:
    params += [(scene.get_stroke_widths(), cfg.lr_width)]
opt = diffvg_utils.SceneOptimizer(scene,
                                  params=params,
                                  num_steps=cfg.num_opt_steps,
                                  lr_min_scale=0.1)



##############################################
# Losses
opt.add_loss('mse', image_losses.MultiscaleMSELoss(rgb=False), cfg.mse_w,
            inputs=('im', 'input_img', 'mse_mul'))
opt.add_loss('bbox', spline_losses.make_bbox_loss(geom.rect(0, 0, w, h)), 1.0,
            inputs=('points',))
if cfg.degree > 3 and cfg.b_spline:
    opt.add_loss('deriv',
                 spline_losses.make_deriv_loss(cfg.deriv, w), cfg.smoothing_w,
                 inputs=('shapes',))

style_loss = image_losses.CLIPPatchLoss(rgb=False, image_prompts=[style_img],
                                        model='CLIPAG', #'ViT-B-32', 
                                        min_size=cfg.patch_size, 
                                        cut_scale=0.0 if cfg.distortion_scale > 0.0 else 0.35, 
                                        distortion_scale=cfg.distortion_scale, 
                                        blur_sigma=0.0, 
                                        thresh=0.0,
                                        n_cuts=64, 
                                        use_negative=False) 

opt.add_loss('style', style_loss, cfg.style_w,
                 inputs=('im',))


##############################################
# Begin visualization and optimize
plut.set_theme()

fig = plt.figure(figsize=(8,8))
gs = GridSpec(3, 2, height_ratios=[1.0, 0.3, 0.3])
gs_sub = GridSpecFromSubplotSpec(1, 2, gs[1,:])

first_frame = None
time_count = 0

render_shapes = []

startup_splines = [path.samples(100).detach().cpu().numpy() for path in scene.shapes]

def frame(step):
    global first_frame, render_shapes, time_count

    perf_t = time.perf_counter()

    opt.zero_grad()
    im = opt.render(background_image)[:,:,0].to(device)

    opt.step(im=im,
             points=scene.get_points(),
             input_img=target_img,
             shapes=scene.shapes,
             mse_mul=cfg.mse_mul)
    
    # Constrain
    with torch.no_grad():
        for path in paths:
            path.param('stroke_width').data.clamp_(cfg.minw, cfg.maxw)
    
    elapsed = time.perf_counter() - perf_t
    time_count += elapsed

    
    must_save = step%cfg.save_every == cfg.save_every-1 or cfg.num_opt_steps==1
    must_show = True
    if cfg.headless and not must_save:
        must_show = must_save

    if must_show:
        # Viz
        im = im.detach().cpu().numpy()
        if first_frame is None:
            first_frame = im

        plt.subplot(gs[0,0])
        plt.title('Startup')
        amt = 0.5
        plt.imshow(target_img*amt + first_frame*(1-amt) , cmap='gray', vmin=0.0, vmax=1.0)
        plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)

        plt.subplot(gs[0,1])
        plt.title('Step %d - elapsed: %.2f'%(step, time_count))
        plt.imshow(background_image, cmap='gray', vmin=0, vmax=1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        plut.setup(box=geom.make_rect(0, 0, w, h))

        plt.subplot(gs_sub[0])
        plt.imshow(style_img, cmap='gray')
        plt.axis('off')

        if opt.losses.has_loss('style'):
            plt.subplot(gs_sub[1])
            plt.imshow(style_loss.test_cutout, cmap='gray')
            plt.axis('off')

        plt.subplot(gs[2,:])
        opt.plot(50)
        
    if saver.valid:
        if must_save:
            print("Saving ", step)
            saver.clear_collected_paths()
            scene.save_json(saver.with_ext('.json'))
            plut.figure_image(adjust=False).save(saver.with_ext('.png'))
            saver.copy_file()
            saver.collected_to_dropbox()


    if cfg.image_movie:
        return im

if cfg.headless:
    filename = ''
else:
    filename = saver.with_ext('.mp4')
plut.show_animation(fig, frame, cfg.num_opt_steps, filename=filename, headless=cfg.headless)
saver.finish()
