from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageFilter
from easydict import EasyDict as edict
import torch
import os
from transformers import get_cosine_schedule_with_warmup

from calligraph import (plut,
                        geom,
                        bspline,
                        bezier,
                        dce,
                        config,
                        util,
                        fs,
                        diffvg_utils,
                        imaging,
                        dslm,
                        stroke_init,
                        segmentation,
                        spline_losses,
                        image_losses,
                        
                        )

import pdb

import torch
device = config.device
dtype = torch.float32

def params():
    size = 350
    padding = 5
    fill = 0
    seed = 133
    alpha =  1 
    image_alpha = 1

    output_path = './generated' #os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/outputs/')

    filename = './data/gml/49867.json'
    
    lr_shape = 2.0
    lr_width = 0.3
    lr_delta_t = 0.05
    lr_delta = 0.1
    lr_Ac = 0 # >0 to optimize also lognormal asymmetry
    num_opt_steps = 300 # 500
    
    smoothing_w = 0.7 
    smoothing_warmup = 0 
    use_clipag = 1

    mse_w = 200.0
    mse_mul = 1 # Factor multiplying each mse blur level (> 1 emph low freq)

    stroke_w = 3
    vary_width = 0

    save_every = 10

    return locals()

cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = '___'
saver = util.SaveHelper(__file__, output_path, use_wandb=cfg.headless,
                        dropbox_folder=cfg.dropbox_folder,
                        cfg=cfg)

S = fs.load_gml(cfg.filename)
S = geom.transform_to_rect(S, geom.make_rect(0, 0, cfg.size, cfg.size), padding=20)

fig = plut.figure_pixels(cfg.size, cfg.size)
plut.stroke(S, 'k', lw=cfg.stroke_w*2)
plut.setup(box=geom.rect(0, 0, cfg.size, cfg.size))
input_img = plut.figure_image(close=True).convert('L')
# Simplify initial paths to retrieve action plan keypoints
startup_paths = [dce.dce(P, 2.5, perimeter=cfg.size) for P in S]

img = np.array(input_img)/255

h, w = img.shape

##############################################
# Settings
verbose = False
np.random.seed(133)

##############################################
# Target tensor
target_img = 1-(1.0-img)*(cfg.alpha*cfg.image_alpha)
target_tensor = torch.tensor(target_img, device=device, dtype=dtype).contiguous()
background_image = np.ones_like(target_img)


scene = diffvg_utils.Scene()

paths = []
fill_color = None


for P in startup_paths:
    m = len(P)-1
    Delta_t = np.ones(m)
    path = dslm.SLMTrajectory(points=P,
                              Delta_t=Delta_t,
                              stroke_width=(cfg.stroke_w, False),
                              Ac=(np.ones(m)*0.01, True),
                              delta=(np.zeros(len(P)-1), True))

    scene.add_shapes([path], stroke_color=([cfg.alpha], False), fill_color=fill_color,
                     split_primitives=True )
    paths.append(path)


# Opt = torch.optim.Adam #Adam
# Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.9), eps=1e-6)

##############################################
# Optimization

params = [(scene.get_points(), cfg.lr_shape),
          (scene.get_params('Delta_t'), cfg.lr_delta_t)]
if cfg.lr_Ac > 0:
    params += [
              (scene.get_params('Ac'), cfg.lr_Ac),
              ]

if cfg.lr_delta > 0:
    params += [
              (scene.get_params('delta'), cfg.lr_delta),
              ]


opt = diffvg_utils.SceneOptimizer(scene,
                                  params=params,
                                  num_steps=cfg.num_opt_steps,
                                  lr_min_scale=0.1)


##############################################
# Losses

opt.add_loss('mse', image_losses.MultiscaleMSELoss(rgb=False), cfg.mse_w,
            inputs=('im', 'input_img', 'mse_mul'))

def smooth_loss(paths):
    return sum([path.time_cost(50)*100 for path in paths])/len(paths)

opt.add_loss('smooth',
               smooth_loss, cfg.smoothing_w,
             ('shapes',))

opt.add_loss('bbox', spline_losses.make_bbox_loss(geom.rect(0, 0, w, h)), 1.0,
            inputs=('points',))

##############################################
# Begin visualization and optimize

plut.set_theme()

fig = plt.figure(figsize=(8,6))
gs = GridSpec(3, 2, height_ratios=[1.0, 0.5, 0.4])

first_frame = None
start_time = None
render_shapes = []

def frame(step):
    global first_frame, render_shapes, start_time

    opt.zero_grad()

    # Rasterize
    with util.perf_timer('render', verbose=verbose):
        im = opt.render(background_image)[:,:,0]

    if cfg.smoothing_warmup > 0:
        smooth_w = (min(step, cfg.smoothing_warmup)/cfg.smoothing_warmup)*cfg.smoothing_w
        opt.losses.replace_weights(smooth=smooth_w)

    opt.step(im=im,
             points=scene.get_points(),
             input_img=target_img,
             shapes=scene.shapes,
             mse_mul=cfg.mse_mul)

    # Constrain
    with torch.no_grad():
        for path in paths:
            #path.param('stroke_width').data.clamp_(cfg.minw, cfg.maxw)
            path.param('Delta_t').data.clamp_(0.1, 0.9)
            path.param('delta').data.clamp_(-np.pi/2, np.pi/2)
            path.param('Ac').data.clamp_(0.0001, 0.1)
            #path.param('stroke_width').data[0] = minw
            #path.param('stroke_width').data[-1] = minw
        
    # Viz
    im = im.detach().cpu().numpy()
    if first_frame is None:
        first_frame = im

    plt.subplot(gs[0,0])
    plt.title('Startup')
    amt = 0.7
    plt.imshow(first_frame*amt + target_img*(1-amt), cmap='gray', vmin=0, vmax=1)

    # for i, path in enumerate(paths):
    #     Q = path.param('points').detach().cpu().numpy()
    #     P = Q[::multiplicity]
    #     plut.stroke(P, 'r', lw=1.3, label='Keypoints' if i==0 else '', alpha=0.5)
    #     plut.stroke(Q, 'c', lw=0.5, label='Control Polygon' if i==0 else '', alpha=0.5)
    plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)


    plt.subplot(gs[0,1])
    plt.title('Step %d'%step)
    amt = 0.8
    plt.imshow(im * amt + target_img*(1-amt), cmap='gray', vmin=0, vmax=1)
    for path in scene.shapes:
        P = path.param('points').detach().cpu().numpy()
        delta = path.param('delta').detach().cpu().numpy()
        dslm.plot_action_plan(P, delta)
    plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)


    plt.subplot(gs[1,:])
    plt.title('Loss')
    opt.plot(50)
    
    ax = plt.subplot(gs[2,:])
    plt.title('Strokes')
    lens = [path.num_points() for path in scene.shapes]
    i = np.argmax(lens)
    with torch.no_grad():
        path_strokes_endt = []
        for path in scene.shapes:
            _, strokes = path.samples(30, get_strokes=True)
            endt = path.endt.detach().cpu()
            path_strokes_endt.append((strokes, endt))
    curt = 0.0
    for strokes, endt in path_strokes_endt:
        dX = np.zeros_like(strokes[0])
        t = np.linspace(curt, curt+endt, len(dX))
        curt += endt
        for V in strokes:
            dX += V
            speed = np.sqrt(V[:,0]**2 + V[:,1]**2)
            plt.fill_between(t, np.zeros_like(speed), speed, color='c', alpha=0.5)
            #plt.plot(speed)
        speed = np.sqrt(dX[:,0]**2 + dX[:,1]**2)
        plt.plot(t, speed, 'k', lw=1)

    if start_time is None:
        start_time = curt

    plut.overlay_bar(ax, curt/start_time)

    if saver.valid:
        
        if step%cfg.save_every == cfg.save_every-1 or cfg.num_opt_steps==1:
            scene.save_json(saver.with_ext('.json'), startup_paths=startup_paths, input_paths=S)
            plut.figure_image().save(saver.with_ext('.png'))
            saver.copy_file()



plut.show_animation(fig, frame, cfg.num_opt_steps, filename=saver.with_ext('.mp4'))
