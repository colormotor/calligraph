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
                        contour_histogram)

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
    return locals()

cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = '___'
saver = util.SaveHelper(__file__, output_path, use_wandb=cfg.headless,
                        dropbox_folder=cfg.dropbox_folder,
                        cfg=cfg)

from py5canvas import Canvas

#S = files.load_gml('./data/janke_gml_1.json') #
S = fs.load_gml('./data/gml/49867.json') #janke_gml_1.json') #[1:2]
#S = files.load_json(os.path.expanduser('~/Dropbox/develop_box/emot_drawing/caz.mp4.json'))
S = geom.transform_to_rect(S, geom.make_rect(0, 0, cfg.size, cfg.size), padding=20)

fig = plut.figure_pixels(cfg.size, cfg.size)
plut.stroke(S, 'k', lw=cfg.stroke_w*2)
input_img = plut.figure_image(close=True)

# c = Canvas(cfg.size, cfg.size)
# c.background(255)
# c.stroke(0)
# c.no_fill()
# c.stroke_weight(cfg.stroke_w*2)
# c.shape(S)
# input_img = c.Image().convert('L')
#P = geom.shapes.star(cfg.size/2.2, center=[cfg.size/2, cfg.size/2])
#P = geom.close(P)
#pad = 50
#P = np.random.uniform(pad, cfg.size-pad, (15, 2))
startup_paths = [dce.dce(P, 2.5, perimeter=cfg.size) for P in S]

img = np.array(input_img)/255

h, w = img.shape

##############################################
# Settings
verbose = False
ref_size = w*cfg.ref_size_factor
offset_variance = [ref_size, ref_size]

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


Opt = torch.optim.Adam #Adam
Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.9), eps=1e-6)


optimizers = [Opt(scene.get_points(), lr=cfg.lr_shape),
              Opt(scene.get_params('Delta_t'), lr=cfg.lr_delta_t)]
if cfg.lr_Ac > 0:
    optimizers += [
              Opt(scene.get_params('Ac'), lr=cfg.lr_Ac),
              ]

if cfg.lr_delta > 0:
    optimizers += [
              Opt(scene.get_params('delta'), lr=cfg.lr_delta),
              ]


warmup = 5
#schedulers = []
schedulers = [get_cosine_schedule_with_warmup(opt, warmup, int(cfg.num_opt_steps*cfg.schedule_decay_factor)) for opt in optimizers]
##############################################
# Losses


losses = util.MultiLoss(verbose=verbose)
#losses.add('mse',
#           image_losses.MultiscaleMSELoss(rgb=False), cfg.mse_w)
mse = image_losses.MultiscaleMSELoss(rgb=False, debug=True)
losses.add('mse',
           mse, cfg.mse_w)

def smooth_loss(paths):
    #return sum([path.endt_norm*100 for path in paths])/len(paths)
    return sum([path.time_cost(50)*100 for path in paths])/len(paths)
    #return sum([path.jerk(10, ref_size=w) for path in paths])/len(paths)

losses.add('smooth',
               smooth_loss, cfg.smoothing_w)

losses.add('bbox',
           spline_losses.make_bbox_loss(geom.make_rect(0, 0, w, h)), 1.0)

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

    for opt in optimizers:
       opt.zero_grad()

    # Rasterize
    with util.perf_timer('render', verbose=verbose):
        im = scene.render(background_image, num_samples=2)[:,:,0].to(device)

    im_no_gamma = im
    if cfg.alpha < 1:
        im = im**300 # # ** gamma

    if cfg.smoothing_warmup > 0:
        smooth_w = (min(step, cfg.smoothing_warmup)/cfg.smoothing_warmup)*cfg.smoothing_w
        losses.replace_weights(smooth=smooth_w)
    # Losses (see weights above)
    loss = losses(
        mse=(im_no_gamma, target_tensor, cfg.mse_mul),
        smooth=(scene.shapes,),
        bbox=(scene.shapes,),
    )

    if cfg.num_opt_steps > 1:
        with util.perf_timer('Opt step', verbose=verbose):
            loss.backward()
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()

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
    #mseim =(mse.target.detach().cpu().numpy() + mse.im.detach().cpu().numpy())/2
    plt.imshow(first_frame*amt + target_img*(1-amt), cmap='gray', vmin=0, vmax=1)
    #plut.stroke(startup_paths, 'r', lw=0.5)
    #plt.imshow(mseim, cmap='gray')

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

    # plt.imshow(background_image, cmap='gray', vmin=0, vmax=1)
    # blur_target = losses.items['mse'][0].blur_target.detach().cpu().numpy()
    # #plt.imshow(np.array(img)/255, cmap='gray', alpha=1)
    # plt.imshow(blur_target, cmap='gray', vmin=0, vmax=1, alpha=0.7)

    # #plut.stroke(startup_path, 'r', lw=1)
    # step_skip = 1

    # if step % step_skip == 0: #with util.perf_timer('Thick curves', verbose=verbose) :
    #     render_shapes = []
    #     for i, path in enumerate(paths):
    #         Q = path.param('points').detach().cpu().numpy()
    #         X = path.samples(len(Q)*10).detach().cpu().numpy()
    #         #X[:,2] *= 0.75
    #         if cfg.fill: # and not overlap:
    #             plut.fill(X[:,:2], 'k', alpha=cfg.alpha*cfg.image_alpha)
    #             plut.stroke(X[:,:2], 'k')
    #         else:
    #             # if not cfg.vary_width:
    #             #     X[:,2] *= 0.5
    #             # S = geom.thick_curve(X, add_cap=False)
    #             # render_shapes.append(S)
    #             plut.stroke(X[:,:2], 'k', lw=cfg.stroke_w)
    #         #plut.fill(S, 'k')

    # if render_shapes:
    #     for S in render_shapes:
    #         plut.fill(S, 'k')

    #plt.legend()
    #plut.setup(box=geom.make_rect(0, 0, w, h))
    plt.subplot(gs[1,:])
    plt.title('Loss')
    for i, (key, kloss) in enumerate(losses.losses.items()):
        if key=='total' or not losses.has_loss(key):
            # There is a bug in 'total'
            continue
        plt.plot(kloss, label='%s:%.4f'%(key, kloss[-1]))
    plt.legend()

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
        save_every = 10
        if step%save_every == save_every-1 or cfg.num_opt_steps==1:
            scene.save_json(saver.with_ext('.json'), startup_paths=startup_paths, input_paths=S)
            plut.figure_image().save(saver.with_ext('.png'))
            saver.copy_file()



plut.show_animation(fig, frame, cfg.num_opt_steps, filename=saver.with_ext('.mp4'))
