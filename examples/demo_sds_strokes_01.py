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
                        walks,
                        files,
                        diffvg_utils,
                        tsp_art,
                        segmentation,
                        imaging,
                        stroke_init,
                        spline_losses,
                        image_losses,
                        contour_histogram)
import pdb

import torch
device = config.device
dtype = torch.float32

def params():
    save = True
    #filename = './data/spock256.jpg' #.jpg'
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/gull.jpg')
    #filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/gauss-1.jpg')
    #filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/woman8.png')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/dog4.jpg')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/normalized.png')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/side-1.jpg')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/sd_woman_2.png')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/sd_color.png')
    # filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/gull2.jpg')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/color.png')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/grey-1.png')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/m1.png')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/miles-1.png')
    filename = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/leslie.png')

    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/kuf3.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/e1.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/stdraw1.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/flo4.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/pat1.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/zcal10.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/music2.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/flo10.jpg')
    style_path = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/flo7.jpg')

    minw, maxw = 0.5, 6.5 #0.75, 4 #5.5  # stroke width range
    degree = 5
    deriv = 3
    multiplicity = 3
    b_spline = 1
    pspline = False
    cardinal = False
    if not b_spline:
        degree = 3
    alpha = 1.0
    closed = False

    lr_shape = 3.0 #5.0 #5.0 #2.0 #2.0 #3.0 # 5.0 # 6 #1.5
    lr_width = 0.5
    num_opt_steps = 300 #300 #150
    vary_width = True
    width_anneal_start = 0.0 #0.4 #
    width_anneal_end = 0.75

    ref_size_factor = 1.0

    single_path = 1
    point_density = 0.003 #5
    # Need to add comma if one element!!!
    targets_and_densities = [
                             (('face_shadow',), point_density*0.7),
                             #(('face_intensity',), point_density*0.7),
                             (('hair',), point_density*1.0),
                             (('leye',), point_density*1.0),
                             (('reye',), point_density*1.0),
                             (('others',), point_density*1.0),
                             (('nose',), point_density*1.0),
                             (('mouth',), point_density*1.0)]
    shadow_range = (0.0, 0.5)
    
    ood = 1
    smoothing_w = 9000.0 #100.0 #0.0 #1000.0 #5000 #3000 #OK #15000 # 500 #50 #10.0 #10.0 20 #1000.0 #1 #1 #1 #500.0 #1 #1000 #3000.0

    target_radius = 30
    curv_w = 0 #10 #1000 #20 # 20.0 #0.0 #5.0 #10.0

    blip = False
    blip_w = 0.1

    clipasso = False
    clip_w = 100.0 #50 #300.0
    lpips_w = 0.0 #10.0 #10.0
    style_w = 0.0 #150.0 #200.0 #60.0 #10 #6 #5 #10 #15 #6.0
    use_clipag = True
    distortion_scale = 0.3 #0.5 #0 #0.3 #5 #5 #0.25 #0.0 #5 #0
    patch_size = 128 #128 # 128 #64

    clip_layer_weights = [(2, 1.0), (3, 1.0)] #, (5, 0.1)]
    #clip_layer_weights = [(2, 1.0), (3, 1.0), (6, 1.0)] #, (1, 0.4)]
    clip_layer_weights = [(2, 0.2), (4, 1.0), (8, 1.0)] #, (1, 0.4)]
    clip_layer_weights = [(2, 1.0), (3, 1.0), (6, 1.0)] #, (6, 1.0)] #, (1, 0.4)]
    clip_layer_weights = [(2, 1.0), (3, 1.0), (4, 1.0)] #, (5, 1.0)] #, (6, 1.0)] #, (6, 1.0)] #, (1, 0.4)]
    #clip_layer_weights = [(9, 1.0)] #, (6, 1.0)]
    #clip_layer_weights = [(2, 1.0), (3, 1.0)] #, (6, 1.0)] #, (6, 1.0)] #, (1, 0.4)]
    # clip_layer_weights = [
    #     (1, 1.0),
    #     (2, 1.0),
    #     #(3, 1.0),
    #     #(9, 1.0)
    # ] #, (3, 1.0)] #, (3, 1.0), (9, 1.0)] #, (5, 1.0)] #, (6, 1.0)] #, (6, 1.0)] #, (1, 0.4)]
    clip_model='CLIPAG'
    clip_model='ViT-B-32-256' #'ViT-L-14' #'ViT-B-16-SigLIP-384' #'CLIPAG' #ViT-B-16' #'CLIPAG'
    #clip_model='ViT-B-32'
    #clip_model='EGCLIP'
    # if 'Sig' in clip_model:
    #     clip_w *= 5

    canny_sigma=1.0 #5
    sds = not clipasso and not blip
    if sds and clipasso:
        clip_w = 100
    sds_w = 1.0
    cond_scale = 0.7 #0.8 #6 #4 #0.7 #9 #4
    guess_mode = True
    if not guess_mode:
        cond_scale = 0.51
    ip_adapter = True
    ip_scale = 0.9 #0.5 #0.2 #0.9 #0.4 #0.9
    cfg = 7.5 #10 #7.5 #14.0 #7.5
    t_min, t_max = 0.5, 0.98
    t_min, t_max = 0.01, 0.98
    t_min, t_max = 0.7, 0.98
    t_min, t_max = 0.1, 0.98
    t_min, t_max = 0.5, 0.98
    t_min, t_max = 0.6, 0.98 #okish
    t_min, t_max = 0.65, 0.98 #okish
    t_min, t_max = 0.7, 0.98 # THICK
    t_min, t_max = 0.6, 0.98 # THICK
    #t_min, t_max = 0.4, 0.98 #GOOD
    #t_min, t_max = 0.5, 0.98 #okish
    #t_min, t_max = 0.6, 0.98 #okish

    grad_method = 'sds'
    #grad_method = 'ism'
    if clipasso:
        t_min, t_max = 0.02, 0.5
    prompt = "Convert to a black and white ink drawing, bold calligraphic strokes"

    overlap_w = 0 #100.0
    blur = 1

    clip_semantic_w = 0.01 #1

    bbox_w = 10.0

    repulsion_subd = 15
    repulsion_w = 0

    C = 2
    start_ang = 0
    orient_w = 0 #100
    startup_w = 1
    mse_w = 0.0
    mse_mul = 1 # Factor multiplying each mse blur level (> 1 emph low freq)

    seed = 333 #1233
    num_iterations = 1

    suffix=''

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
style_img = Image.open(cfg.style_path).convert('L').resize((size, size))

img = np.array(input_img)/255

cond_img = feature.canny(img, cfg.canny_sigma)
cond_img = Image.fromarray((cond_img*255).astype(np.uint8)).convert('RGB')
saver.log_image('Cond image', cond_img)

h, w = img.shape
box = geom.make_rect(0, 0, w, h)

##############################################
# Settings
verbose = False

overlap = False
ref_size = w
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

from calligraph.portrait import Portrait
portrait = Portrait(saliency_img, shadow_range=cfg.shadow_range, face_intensity_range=(0.44, 0.55))
scene = diffvg_utils.Scene()
density_maps = []
for feats, point_dens in cfg.targets_and_densities:
    target = portrait.get_targets(*feats, merge=True)
    nump = max(int(np.sum(1-target)*point_dens), 3)
    startup_paths, density = stroke_init.init_path_diffusion(target, nump, cfg.startup_w, padding=20)
    if not startup_paths:
        print("No paths for ", feats)
        continue
    startup_paths = [add_multiplicity(P) for P in startup_paths]
    density_maps.append(density)
    for Pw in startup_paths:
        if cfg.b_spline:
            path = diffvg_utils.DynamicBSpline(Pw[:,:2],
                                            stroke_width=(Pw[:,2], True),
                                            degree=cfg.degree,
                                            pspline=cfg.pspline,
                                            multiplicity=cfg.multiplicity,
                                            split_pieces=cfg.alpha < 1 and cfg.overlap_w > 0,
                                            #init_smooth_params=dict(r=1.0,
                                            #                    der=3) if cfg.multiplicity > 1 else {},
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

density_map = sum(density_maps)/len(density_maps)

Opt = torch.optim.Adam #Adam
Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.999)) #, eps=1e-6)
#Opt = lambda params, lr: torch.optim.Adam(params, lr) #, betas=(0.7, 0.9), eps=1e-6)

optimizers = [Opt(scene.get_points(), lr=cfg.lr_shape)]
if cfg.vary_width:
    optimizers += [Opt(scene.get_stroke_widths(), lr=cfg.lr_width)]

schedulers = [util.step_cosine_lr_scheduler(opt, 0.0, 0.2, cfg.num_opt_steps) for opt in optimizers]
#schedulers = [util.step_cosine_lr_scheduler(opt, 0.5, 0.1, cfg.num_opt_steps) for opt in optimizers]
#schedulers = [util.step_cosine_lr_scheduler(opt, 0.5, 0.2, cfg.num_opt_steps) for opt in optimizers]
#schedulers = []


#initialize(background_image)

##############################################
# Losses


losses = util.MultiLoss(verbose=verbose)
losses.add('mse',
           image_losses.MultiscaleMSELoss(rgb=False), cfg.mse_w)
if cfg.b_spline: # b_spline:
    losses.add('deriv',
               spline_losses.make_deriv_loss(cfg.deriv, ref_size), cfg.smoothing_w)

# loss
# es.add('repulsion',
#                spline_losses.make_repulsion_loss(cfg.repulsion_subd, False, signed=True), cfg.repulsion_w)

losses.add('curv',
               spline_losses.make_curvature_loss(cfg.target_radius, multiplicity=cfg.multiplicity, absolute=True), cfg.curv_w)

losses.add('orient',
    spline_losses.make_orientation_loss(cfg.multiplicity,
                                               cfg.C, cfg.start_ang, normalize=True), cfg.orient_w)

losses.add('bbox',
           spline_losses.make_bbox_loss(geom.make_rect(0, 0, w, h)), cfg.bbox_w)

if cfg.blip:
    from transformers import BlipProcessor, BlipModel
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # processor = Blip3Processor.from_pretrained("Salesforce/blip3-ocr-200m")
    # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip3-ocr-200m").to(device)
    cls = False
    # COSINE seemse to be significantly better
    # 2, 5, 9 is ok
    #
    blip_loss = image_losses.VisionEncoderLoss(model, processor,
                                              layer_weights=[

                                                  #(-1, 1.0),
                                                  #(-2, 1.0),
                                                  #(-3, 1.0),
                                                  #(0, 1.0),
                                                  (2, 1.0), # OK?
                                                  #(4, 1.0), # OK?
                                                  #(5, 1.0), # !
                                                  #(6, 1.0), #

                                                  # (8, 1.0), #OK
                                                  #(7, 1.0), #OK
                                                  (9, 1.0), #OK
                                                  #(10, 1.0), #OK

                                                  #(12, 1.0), #? with 3
                                                  #(3, 1.0), #OK
                                                 #  (9, 1.0), #OK
                                                 #  (10, 1.0), #OK

                                                             ],
                                               cls=cls,
                                               pool_no_cls=False,
                                               distortion_scale=0.1,
                                               metric='cosine', #'cosine', #'L1', #'cosine', #l2',
                                               rgb=False)

    losses.add('blip',
               blip_loss, cfg.blip_w)

if cfg.clipasso:
    clip_loss = image_losses.CLIPVisualLoss(rgb=False,
                                            clip_model=cfg.clip_model,
                                            #clip_model='CLIPAG',
                                            semantic_w=cfg.clip_semantic_w,
                                            distortion_scale=0.5, #0.3, #15,
                                            vis_metric='L1', #cosine',
                                            crop_scale=(0.9, 1.0), #(0.5, 1.0),
                                            layer_weights=cfg.clip_layer_weights)
    losses.add('clip',
               clip_loss, cfg.clip_w)

if cfg.lpips_w > 0:
    lpips = image_losses.LPIPS(rgb=False)
    #lpips = image_losses.VGGPerceptualLoss(rgb=False)
    losses.add('lpips', lpips, cfg.lpips_w)

if cfg.sds:
    if cfg.style_w > 0:
        sd.cfg.enable_sequential_cpu_offload = True
    sds = sd.SDSLoss(cfg.prompt,
                     augment=0,
                     rgb=False,
                     controlnet="lllyasviel/sd-controlnet-canny", # if not cfg.ip_adapter else '',
                     #controlnet="lllyasviel/sd-controlnet-scribble",
                     seed=cfg.seed, #777, #999,
                    t_range=[cfg.t_min, cfg.t_max],
                    guidance_scale=cfg.cfg,
                     conditioning_scale=cfg.cond_scale, #9,
                    num_hifa_denoise_steps=4,
                    ip_adapter='ip-adapter-plus_sd15.bin' if cfg.ip_adapter else '',
                     ip_adapter_scale=cfg.ip_scale, #1.3, #0.9,

                     time_schedule='ism', #'linear', #'ism' if cfg.grad_method == 'ism' else 'linear', #ism', #'dtc', #'dtc', #'pow', #dtc', #linear', #'dtc', #'random', #'dreamtime', #'dtc', #'dreamtime', #'pow',
                    grad_method=cfg.grad_method, #ism', #csd', #ism', #'ism', #'ism', #'sds',
                     guess_mode=cfg.guess_mode)
    def sds_loss(im, step):
        return sds(im,
                   cond_img, # if not cfg.ip_adapter else None,
                   step, cfg.num_opt_steps,
                   grad_scale=0.01 if sds.grad_method == 'ism' else 0.1,
                   ip_adapter_image=input_img
                )
    losses.add('sds',
               sds_loss, cfg.sds_w)

#semantic_loss = image_losses.CLIPAGSemanticLoss('Calligraphy flourishes', rgb=False, use_negative=True)
#sem_loss = image_losses.CLIPVisualLoss(rgb=False, clipag=cfg.use_clipag, semantic_w=1, geometric_w=0.0) #cfg.clip_semantic_w)
# patch_loss = image_losses.CLIPPatchLoss(rgb=False, image_prompts=[style_img],
#                                         min_size=128,
#                                         model='CLIPAG', use_negative=False) #clipag=cfg.use_clipag)
style_loss = image_losses.CLIPPatchLoss(rgb=False, image_prompts=[style_img],
                                        model='CLIPAG', #'ViT-B-16', #'ViT-B-32-256', #'ViT-SO400M-14-SigLIP', #'ViT-L-14', #'CLIPAG',
                                        min_size=cfg.patch_size, #128, #64, #64, #64, #128, #32, #128, #64, #64, #100, #64, #128,
                                        cut_scale=0.0 if cfg.distortion_scale > 0.0 else 0.35, #5, #0.5, #0.1,
                                        distortion_scale=cfg.distortion_scale, #0.5, #5,
                                        blur_sigma=0.0, #1.0,
                                        thresh=0.0,
                                        n_cuts=64, #32, #64, #32, #64,
                                        num_batches=1,
                                        use_negative=False) #, n_cuts=24) #16)

losses.add('style',
               style_loss, cfg.style_w)

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
    for opt in optimizers:
       opt.zero_grad()

    # Rasterize
    with util.perf_timer('render', verbose=verbose):
        im = scene.render(background_image, num_samples=2)[:,:,0].to(device)

    im_no_gamma = im
    if cfg.alpha < 1.0:
        im = im**100

    minw = cfg.minw
    if cfg.width_anneal_start > 0 and step > cfg.num_opt_steps*cfg.width_anneal_start:
        #minw = 0.0
        start =  cfg.num_opt_steps*cfg.width_anneal_start
        end = cfg.num_opt_steps*cfg.width_anneal_end
        t = np.clip((step - start)/(end - start), 0.0, 1.0)
        minw = minw - t*minw
        print('annealing w', minw)

    # anneal_start = cfg.num_opt_steps/2
    # if step > anneal_start:
    #     anneal_dur = cfg.num_opt_steps - anneal_start
    #     t = 1-((step-anneal_start)/anneal_dur)
    #     tmin = 0.1
    #     #tmin = tmin + (cfg.t_min - tmin)*t
    #     #print("Annealing tmin:", tmin, t)
    #     sds.t_range[0] = tmin

    # Losses (see weights above)
    loss = losses(
        mse=(im, target_tensor, cfg.mse_mul),
        clip=(im, target_tensor),
        lpips=(im, target_tensor),
        blip=(im, target_tensor, 4,),
        semantic=(im,),
        sds=(im, step),
        tv=(im,),
        offset=(scene.shapes,),
        approx=(scene.shapes,),
        curv=(scene.shapes,),
        orient=(scene.shapes,),
        deriv=(scene.shapes,),
        overlap=(im_no_gamma, scene.shapes,),
        repul=(scene.shapes,),
        bbox=(scene.shapes,),
    )

    with util.perf_timer('Opt step', verbose=verbose):
        loss.backward()
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

    # Constrain
    # Constrain
    with torch.no_grad():
        for path in scene.shapes:
            path.param('stroke_width').data.clamp_(minw, cfg.maxw)
            #path.param('stroke_width').data[0] = minw
            #path.param('stroke_width').data[-1] = minw

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed

    save_every = 10
    must_save = step%save_every == save_every-1 or cfg.num_opt_steps==1
    must_show = True
    # if cfg.headless and not must_save:
    #     must_show = must_save

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
        plt.imshow(first_frame*amt + target_img*(1-amt) , cmap='gray')
        # for i, path in enumerate(paths):
        #     Q = path.param('points').detach().cpu().numpy()
        #     P = Q[::multiplicity]
        #     plut.stroke(P, 'r', lw=1.3, label='Keypoints' if i==0 else '', alpha=0.5)
        #     plut.stroke(Q, 'c', lw=0.5, label='Control Polygon' if i==0 else '', alpha=0.5)
        plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)

        plt.subplot(gs[0,1])
        lrs = 'lr %.3f'%(optimizers[0].param_groups[0]['lr'])
        # if losses.has_loss('sds'):
        #     plt.title('Step %d, t %d, %s' %(step, int(sds.t_saved), lrs))
        # else:
        #     plt.title('Iter: %d Step %d, %s'%(iteration, step, lrs))
        plt.imshow(background_image, cmap='gray', vmin=0, vmax=1)

        if losses.has_loss('curv'):
            plut.fill_circle([cfg.target_radius, cfg.target_radius], cfg.target_radius, 'c')
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)

        plt.subplot(gs_sub[0,0])
        if cfg.clipasso:
            plt.imshow(clip_loss.y_augs[1])
        if cfg.sds:
            plt.imshow(np.array(cond_img))
        # plt.subplot(gs_sub[0,1])
        # plt.imshow(density_map)
        plt.subplot(gs_sub[0,2])
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
        if losses.has_loss('sds'):
            plt.title('Step %d, t %d, %s, %.2f' %(step, int(sds.t_saved), lrs, time_count))
        else:
            plt.title('Iter: %d Step %d, %s'%(iteration, step, lrs))
        for i, (key, kloss) in enumerate(losses.losses.items()):
            if key=='total' or not losses.has_loss(key):
                # There is a bug in 'total'
                continue
            plt.plot(kloss, label='%s:%.4f'%(key, kloss[-1]))
        plt.legend()
        plt.tight_layout()

    if saver.valid and must_save and step > 100:
        print('Saving')
        saver.clear_collected_paths()
        scene.save_json(saver.with_ext('.json'))
        cfg.save_yaml(saver.with_ext('.yaml'))
        plut.figure_image(adjust=False).save(saver.with_ext('.png'))
        saveim = Image.fromarray((im*255).astype(np.uint8))
        saveim.save(saver.with_ext('.png', suffix='output'))
        #saver.log_image('output', plt.gcf())
        saver.copy_file()
        saver.collected_to_dropbox()


if False: #cfg.headless:
    filename = ''
else:
    filename = saver.with_ext('.mp4')
plut.show_animation(fig, frame, cfg.num_opt_steps, filename=filename, headless=cfg.headless)
saver.finish(filename)
