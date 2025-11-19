from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image
from easydict import EasyDict as edict
import torch
from skimage import feature
from transformers import get_cosine_schedule_with_warmup
import time, os
from calligraph import (plut,
                        geom,
                        bspline,
                        bezier,
                        dce,
                        sd,
                        config,
                        util,
                        fs,
                        diffvg_utils,
                        tsp_art,
                        segmentation,
                        imaging,
                        stroke_init,
                        spline_losses,
                        image_losses,)
import pdb

import torch
device = config.device
dtype = torch.float32

def params():
    # Script parameters
    # These are automatically converted to named script arguments
    # Currently requires booleans to be 0 or 1 to work from cmd-line
    save = True
    output_path = './generated/'
    output_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/outputs/'
    
    filename = './data/spock.jpg'
    #filename = './data/dog4.jpg'
    style_path = './data/style_imgs/music.jpg'
    #style_path = './data/style_imgs/zcal2.jpg' 
    
    mask = None #'./data/utah-mask.jpg' #utah-fix.jpg' #.jpg'

    minw, maxw = 0.5, 7.0 
    degree = 5
    deriv = 3
    multiplicity = 3
    b_spline = 1      
    pspline = False  # If 1 use penalized spline smoothing approach
    cardinal = False # If 1 uses Catmull-Rom splines (no smoothing)
    if not b_spline:
        degree = 3
    alpha = 1.0
    closed = False

    ref_size_factor = 1.0
    point_density = 0.002 #3 #5
    startup_width = 1.0

    lr_pos = 3.0 
    lr_width = 0.5
    num_opt_steps = 300 #300 #150
    vary_width = True
    width_anneal_start = 0.0 #0.4 # Use > 0 to anneal stroke widths to zero
    width_anneal_end = 0.75

    ood = 1
    smoothing_w = 200.0 

    clipasso = False
    clip_w = 100.0 
    lpips_w = 0.0 
    style_w = 200.0 
    distortion_scale = 0.3 
    patch_size = 128 

    clip_layer_weights = [(2, 1.0), (3, 1.0), (4, 1.0)] #, (6, 1.0)] #, (1, 0.4)]
    clip_model='CLIPAG'
    clip_semantic_w = 0.0

    canny_sigma=1.0 #5
    sds = not clipasso
    if sds and clipasso:
        clip_w = 100
        
    sds_w = 1.0
    cond_scale = 0.7 #0.8 #6 #4 #0.7 #9 #4
    guess_mode = True
    if not guess_mode:
        cond_scale = 0.51
    ip_adapter = True
    ip_scale = 0.9 
    cfg = 7.5 

    t_min, t_max = 0.5, 0.98 

    grad_method = 'ism' #either 'sds' or 'ism'
    
    prompt = "A single line pen drawing, single stroke, thin line"
    prompt = "An architectural drawing"
    prompt = "A pencil sketch, thin strokes"
    prompt = "A black and white ink drawing"
    
    bbox_w = 10.0

    mse_w = 0.0
    mse_mul = 1 # Factor multiplying each mse blur level (> 1 emph low freq)

    seed = 333 
    num_iterations = 1

    suffix=''

    image_movie = True

    return locals()

cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = '___'
if cfg.sds:
    cfg.suffix += '_sds'
else:
    cfg.suffix += '_clip'

saver = util.SaveHelper(__file__, output_path, 
                        suffix=cfg.suffix,
                        cfg=cfg)


size = 512
input_img = Image.open(cfg.filename).convert('L').resize((size, size))
saver.log_image('Input image', input_img)


style_img = Image.open(cfg.style_path).convert('L').resize((size, size))

img = np.array(input_img)/255

cond_img = feature.canny(img, cfg.canny_sigma)
cond_img = Image.fromarray((cond_img*255).astype(np.uint8)).convert('RGB')


h, w = img.shape
box = geom.make_rect(0, 0, w, h)

##############################################
# Settings
verbose = False

overlap = False
ref_size = w/cfg.ref_size_factor
offset_variance = [ref_size, ref_size]

diffvg_utils.cfg.one_channel_is_alpha = True
np.random.seed(cfg.seed)

##############################################
# Target tensor
target_img = img
target_tensor = torch.tensor(target_img, device=device, dtype=dtype).contiguous()
background_image = np.ones_like(target_img)

def add_multiplicity(Q, noise=0.0):
    Q = np.kron(Q, np.ones((cfg.multiplicity, 1)))
    return Q + np.random.uniform(-noise, noise, Q.shape)

##############################################
# Initialization paths

from scipy.ndimage import gaussian_filter


def process_image(img):
    return np.minimum(gaussian_filter(1-img, 6)*4, 1.0)

num_points = int(np.sum(1-img)*cfg.point_density)
startup_paths, density_map = stroke_init.init_path_tsp(input_img,
                                                       num_points,
                                                       startup_w=cfg.startup_width,
                                                       saliency_type='ood')

if cfg.b_spline or cfg.cardinal:
    startup_paths = [add_multiplicity(P) for P in startup_paths]

scene = diffvg_utils.Scene()

for Pw in startup_paths:
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(Pw[:,:2],
                                        stroke_width=(Pw[:,2], True),
                                        degree=cfg.degree,
                                        pspline=cfg.pspline,
                                        multiplicity=cfg.multiplicity,
                                        closed=cfg.closed)
    else:
        if cfg.cardinal:
            path = diffvg_utils.CardinalSpline(Pw[:,:2],
                                        stroke_width=(Pw[:,2], True),
                                        closed=cfg.closed)
        else:
            Q = bezier.cubic_bspline_to_bezier_chain(Pw, periodic=cfg.closed)
            if cfg.closed:
                Q = Q[:-1]
            path = diffvg_utils.Path(Q[:,:2],
                                    degree=3,
                                    stroke_width=(Q[:,2], True),
                                    closed=cfg.closed)
    scene.add_shapes([path], stroke_color=([cfg.alpha], True), fill_color=None, split_primitives=True)



##############################################
# Optimization

params = [(scene.get_points(), cfg.lr_pos)]
if cfg.vary_width:
    params += [(scene.get_stroke_widths(), cfg.lr_width)]
opt = diffvg_utils.SceneOptimizer(scene,
                                  params=params,
                                  num_steps=cfg.num_opt_steps,
                                  lr_min_scale=0.2)



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


if cfg.clipasso:
    clip_loss = image_losses.CLIPVisualLoss(rgb=False,
                                            clip_model=cfg.clip_model,
                                            #clip_model='CLIPAG',
                                            semantic_w=cfg.clip_semantic_w,
                                            distortion_scale=0.5, #0.3, #15,
                                            vis_metric='L1', #cosine',
                                            crop_scale=(0.9, 1.0), #(0.5, 1.0),
                                            layer_weights=cfg.clip_layer_weights)
    opt.add_loss('clip',
               clip_loss, cfg.clip_w, ('im', 'input_img'))

if cfg.sds:
    # if cfg.style_w > 0:
    #     sd.cfg.enable_sequential_cpu_offload = True
    sds = sd.SDSLoss(cfg.prompt,
                     augment=0,
                     rgb=False,
                     controlnet="lllyasviel/sd-controlnet-canny", # if not cfg.ip_adapter else '',
                     #controlnet="lllyasviel/sd-controlnet-scribble",
                     seed=cfg.seed, 
                    t_range=[cfg.t_min, cfg.t_max],
                    guidance_scale=cfg.cfg,
                     conditioning_scale=cfg.cond_scale, #9,
                    num_hifa_denoise_steps=4,
                    ip_adapter='ip-adapter-plus_sd15.bin' if cfg.ip_adapter else '',
                     ip_adapter_scale=cfg.ip_scale, 
                     time_schedule='ism', 
                    grad_method=cfg.grad_method, 
                     guess_mode=cfg.guess_mode)
    def sds_loss(im, step):
        return sds(im,
                   cond_img, 
                   step, cfg.num_opt_steps,
                   grad_scale=0.01 if sds.grad_method == 'ism' else 0.1,
                   ip_adapter_image=input_img
                )
    opt.add_loss('sds',
                 sds_loss, cfg.sds_w, ('im', 'step'))

style_loss = image_losses.CLIPPatchLoss(rgb=False, image_prompts=[style_img],
                                        model=cfg.clip_model,
                                        min_size=cfg.patch_size, 
                                        cut_scale=0.0 if cfg.distortion_scale > 0.0 else 0.35, 
                                        distortion_scale=cfg.distortion_scale, 
                                        blur_sigma=0.0, 
                                        thresh=0.0,
                                        n_cuts=64, 
                                        num_batches=1,
                                        use_negative=False) 


opt.add_loss('style',
               style_loss, cfg.style_w, ('im',))

##############################################
# Begin visualization and optimize

plut.set_theme()

fig = plt.figure(figsize=(8,8))
gs = GridSpec(3, 2, height_ratios=[1.0, 0.5, 0.5])
gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :])

first_frame = None
current_img = np.array(background_image)

time_count = 0
iteration = 0

def frame(step):
    global first_frame, time_count, iteration, background_image


    perf_t = time.perf_counter()
    opt.zero_grad()
    
    # Rasterize
    with util.perf_timer('render', verbose=True): #verbose):
        im = opt.render(background_image)[:,:,0].to(device)

    minw = cfg.minw
    if cfg.width_anneal_start > 0 and step > cfg.num_opt_steps*cfg.width_anneal_start:
        start =  cfg.num_opt_steps*cfg.width_anneal_start
        end = cfg.num_opt_steps*cfg.width_anneal_end
        t = np.clip((step - start)/(end - start), 0.0, 1.0)
        minw = minw - t*minw
        print('annealing w', minw)

    opt.step(im=im,
             points=scene.get_points(),
             input_img=input_img,
             shapes=scene.shapes,
             step=step,
             mse_mul=cfg.mse_mul)

    # Constrain
    with torch.no_grad():
        for path in scene.shapes:
            path.param('stroke_width').data.clamp_(minw, cfg.maxw)

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed

    save_every = 10
    must_save = step%save_every == save_every-1 or cfg.num_opt_steps==1
    must_show = True

    if must_show:

        # Viz
        im = im.detach().cpu().numpy()
        if first_frame is None:
            first_frame = im

        if saver.valid:
            plt.suptitle(saver.name)
        plt.subplot(gs[0,0])
        plt.title('Startup - time: %.3f'%(time_count))
        amt = 0.5
        plt.imshow(im*amt + target_img*(1-amt) , cmap='gray')
        # for i, path in enumerate(paths):
        #     Q = path.param('points').detach().cpu().numpy()
        #     P = Q[::multiplicity]
        #     plut.stroke(P, 'r', lw=1.3, label='Keypoints' if i==0 else '', alpha=0.5)
        #     plut.stroke(Q, 'c', lw=0.5, label='Control Polygon' if i==0 else '', alpha=0.5)
        plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)

        plt.subplot(gs[0,1])
        lrs = 'lr %.3f'%(opt.optimizers[0].param_groups[0]['lr'])
        if opt.has_loss('sds'):
            plt.title('Step %d, t %d, %s' %(step, int(sds.t_saved), lrs))
        else:
            plt.title('Iter: %d Step %d, %s'%(iteration, step, lrs))
        plt.imshow(background_image, cmap='gray', vmin=0, vmax=1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)

        plt.subplot(gs_sub[0,0])
        if cfg.clipasso:
            plt.title('Clip aug')
            plt.imshow(clip_loss.y_augs[1])
        if cfg.sds:
            plt.title('Cond image')
            plt.imshow(np.array(cond_img))
        plt.subplot(gs_sub[0,1])
        plt.title('Saliency')
        plt.imshow(density_map)
        plt.subplot(gs_sub[0,2])
        plt.title(f'Style, w:{cfg.style_w}')
        plt.imshow(np.array(style_img), cmap='gray')
        # with util.perf_timer('Thick curves', verbose=verbose):
        #     for i, path in enumerate(scene.shapes):
        #         Q = path.param('points').detach().cpu().numpy()
        #         X = path.samples(len(Q)*10).detach().cpu().numpy()
        #         #X[:,2] *= 0.75
        #         if cfg.closed: # and not overlap:
        #             fill = [0.0, 0.0, 0.0, 0.1] #scene.shape_groups[i].fill_color.detach().cpu().numpy()
        #             a = fill[-1]
        #             plut.fill(X[:,:2], fill[:3], alpha=a)
        #             plut.stroke(X[:,:2], 'k')
        #         else:
        #             S = geom.thick_curve(X, add_cap=False)
        #             plut.fill(S, 'k')

        # plut.setup(box=geom.make_rect(0, 0, w, h))
        plt.subplot(gs[2,:])
        if opt.has_loss('sds'):
            plt.title('Step %d, t %d, %s, %.2f' %(step, int(sds.t_saved), lrs, time_count))
        else:
            plt.title('Iter: %d Step %d, %s'%(iteration, step, lrs))
        opt.plot(50)
        plt.tight_layout()

    if saver.valid and must_save and (step > 100 or cfg.num_opt_steps==1):
        print('Saving')
        saver.clear_collected_paths()
        scene.save_json(saver.with_ext('.json'), startup_paths=startup_paths)
        cfg.save_yaml(saver.with_ext('.yaml'))
        plut.figure_image(adjust=False).save(saver.with_ext('.png'))
        saveim = Image.fromarray((im*255).astype(np.uint8))
        saveim.save(saver.with_ext('.png', suffix='output'))
        saver.copy_file()

    if cfg.image_movie:
        return im

filename = saver.with_ext('.mp4')
plut.show_animation(fig, frame, cfg.num_opt_steps, filename=filename, headless=cfg.headless)
saver.finish(filename)
