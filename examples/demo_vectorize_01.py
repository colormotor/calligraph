from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, ImageFilter
from easydict import EasyDict as edict
import torch
from skimage import feature
import torch.nn.functional as F
import os

from calligraph import (plut,
                        geom,
                        bspline,
                        bezier,
                        clipper,
                        dce,
                        sd,
                        config,
                        util,
                        fs,
                        ase_palette,
                        diffvg_utils,
                        tsp_art,
                        segmentation,
                        imaging,
                        stroke_init,
                        spline_losses,
                        image_losses)

import time
device = config.device
dtype = torch.float32

# Color conversion utils
def rgb2lab(im):
    print(im.shape)
    im = to_batch(im)
    im = rgb_to_lab(im).permute((2, 3, 1, 0))
    res = im.squeeze(0).squeeze(-1)
    print('2lab', res.shape)
    return res

def lab2rgb(im):
    print(im.shape)
    im = to_batch(im)
    im = lab_to_rgb(im).permute((2, 3, 1, 0))
    res = im.squeeze(0).squeeze(-1)
    print('2rgb', res.shape)
    return res

def to_batch(im):
    im = im.unsqueeze(0)
    im = im.unsqueeze(-1)
    im = im.permute((3, 2, 0, 1))
    return im

def to_palette(im, n):
    im = im.quantize(n, method=Image.Quantize.MEDIANCUT, kmeans=n).convert('RGB')
    colors = im.getcolors()
    return [np.array(c[1])/255 for c in colors]

# Config
def params():
    output_path = './outputs'
    save = True
    filename = './data/bach.jpg'

    minw, maxw = 0.5, 2  # stroke width range
    degree = 5 # Spline degree
    deriv = 3  # Smoothing degrees
    multiplicity = 1 # Keypoint multiplicity
    b_spline = 1
    pspline = 0 # If 1 (true) use discrete penalized spline_points

    num_voronoi_points = 50
    ds = 5
    lr_shape = 2.0
    lr_min_scale = 0.7
    lr_color = 1e-2
    num_opt_steps = 300

    use_color = 0

    clip_semantic_w = 0.0
    clip_model = 'CLIPAG'
    clip_layer_weights = [(2, 1.0), (3, 1.0), (6, 1.0)]
    clipasso = True
    clip_w = 300.0
    semantic_w = 0.0

    sds = not clipasso
    sds_w = 1.0
    t_min, t_max = 0.45, 0.98
    cond_scale=0.51
    grad_method = 'sds'
    grad_method = 'ism'

    smoothing_w = 0.5

    lab = False
    K = 3
    chans = 3
    if chans != 3:
        lab = False
    palette = torch.linspace(0, 1, K+1, device=device)[:-1]
    palette = torch.linspace(0, 1, K, device=device)
    palette = torch.vstack([palette]*chans).T

    palette_file = './data/palettes/ 5
    pfiles = fs.files_in_dir('/home/danielberio/Dropbox/transfer_box/data/calligraph/palettes', 'ase')
    palette_file = pfiles[8]
    palette_file = '/home/danielberio/Dropbox/transfer_box/data/calligraph/palettes/Jamba Juice.ase' #Honey Pot.ase'
    palette_file = '/home/danielberio/Dropbox/transfer_box/data/calligraph/palettes/rocket021x.ase'
    palette_file = '/home/danielberio/Dropbox/transfer_box/data/calligraph/palettes/Metropolitan.ase' #Honey Pot.ase'
    palette = ase_palette.load(palette_file)[0]

    palette_im = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/pat5.jpg')
    palette_im = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/hk4.jpg')
    palette_im = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/camo-blue.jpg')
    palette_im = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/palettes/swirl3.png')
    palette_im = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/camo8.jpg')
    palette_im = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/camo11.jpg')

    gumbel_hard = 0
    tau_start = 1.0 #3.0

    style_w = 30.0 # 20.0 #30.0 #10.0 #7.0 #3.0 #7.0 #1.4 #0.7 #0.1 #0.75 #1.0 #2 #1 # 0.5
    style_path = '' #palette_im # os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/camo8.jpg')
    stroke_w = 0.0 #1.5 #0.5 #5 #0.5
    stroke_darkness = 0.5

    repulsion_subd = 10 # 15 #0 #15 #15 #10 #5
    repulsion_w = 5000 #100 # 1000 #7000 #1000 #15000
    resolve_self_ins_every = 0 #5 #1
    repulsion_d = 10

    rand_init = 2

    mse_w = 0.0
    mse_mul = 1 # Factor multiplying each mse blur level (> 1 emph low freq)

    seed = 1233
    suffix = ''
    return locals()

cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = '___'
saver = util.SaveHelper(__file__, output_path, use_wandb=False, #cfg.headless,
                        dropbox_folder=cfg.dropbox_folder,
                        suffix=cfg.suffix,
                        cfg=cfg)

cfg.palette = to_palette(Image.open(cfg.palette_im), 7) # 2) #7)


cfg.palette = torch.tensor(cfg.palette, device=device, dtype=torch.float32)
cfg.K = len(cfg.palette)
cfg.gumbel_scale = 0.15 #15 #0.15

if cfg.lab:
    cfg.palette = rgb2lab(cfg.palette)

#filename = './data/utah.jpg'
input_img = Image.open(cfg.filename).resize((400, 400)) #512, 512))
saver.log_image('Input image', input_img)

img = np.mean(np.array(input_img)/255, axis=-1)

style_im = cfg.palette_im

if cfg.style_path: # is not None:
    style_im = cfg.style_path
style_img = Image.open(style_im).convert('RGB')

cond_img = feature.canny(img, 1.0)
cond_img = Image.fromarray((cond_img*255).astype(np.uint8)).convert('RGB')
saver.log_image('Cond image', cond_img)

h, w = img.shape
box = geom.make_rect(0, 0, w, h)

##############################################
# Settings
verbose = False

overlap = False
closed = True

diffvg_utils.cfg.one_channel_is_alpha = False
np.random.seed(cfg.seed)

##############################################
# Target tensor
target_img = np.array(input_img)/255
target_tensor = torch.tensor(target_img, device=device, dtype=dtype).contiguous()
background_image = np.ones_like(target_img)

def add_multiplicity(Q, noise=0.0):
    Q = np.kron(Q, np.ones((cfg.multiplicity, 1)))
    return Q + np.random.uniform(-noise, noise, Q.shape)

##############################################
# Initialization paths

# Saliency
from calligraph import ood_saliency
sal = ood_saliency.compute_saliency(input_img.convert('RGB'))[0]/255
sal *= segmentation.clip_saliency(input_img)
#sal = np.ones((h, w))
density_map = ((sal-sal.min())/(sal.max() - sal.min()))**2 #(sal*(1-img))

# sal_file = os.path.splitext(cfg.filename)[0]+'-sal.jpg'
# if os.path.isfile(sal_file):
#     density_map = np.array(Image.open(sal_file).convert('L').resize((w, h)))/255
# Voronoi regions
points, verts = tsp_art.weighted_voronoi_sampling(density_map, cfg.num_voronoi_points, get_regions=True, nb_iter=50)
verts = [geom.uniform_sample(V, cfg.ds*2, closed=True) for V in verts]
verts = [V + np.random.uniform(-1,1,V.shape)*cfg.rand_init for V in verts]

# Sort by saliency
rast = imaging.ShapeRasterizer(box, w)
im = np.array(img)/255

colors = []
saliencies = []
palette = cfg.palette.detach().cpu().numpy()
for V in verts:
    rast.clear()
    rast.fill_shape(V)
    mask = np.array(rast.image()).astype(bool)
    s = np.mean(density_map[mask])
    v = np.mean(img[mask], axis=0)
    saliencies.append(s)
    colors.append(v)
    #dists = np.linalg.norm(palette - v, axis=1)
    #initial_logits.append(np.mean(dists) - dists)
    #initial_logits.append(-(dists - dists.mean()) / (dists.std() + 1e-8))
#initial_logits.append(np.zeros_like(initial_logits[-1])) # Background

# Sort by increasing salieny
I = np.argsort(saliencies)
colors = [colors[i] for i in I]
startup_paths = [verts[i] for i in I]
startup_paths = [add_multiplicity(P) for P in startup_paths]

scene = diffvg_utils.Scene()

closed = True

for P, clr in zip(startup_paths, colors):
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(P,
                degree=cfg.degree,
                stroke_width=(cfg.stroke_w, False),
                pspline=cfg.pspline,
                closed=closed)
    else:
        # Q = bezier.cubic_bspline_to_bezier_chain(P, periodic=closed)
        # if closed:
        #     Q = Q[:-1]
        # path = diffvg_utils.Path(Q[:,:2],
        #                          degree=3,
        #                          stroke_width=(cfg.stroke_w, False),
        #                          closed=closed)
        path = diffvg_utils.CardinalSpline(P,
                                           stroke_width=(cfg.stroke_w, False),
                                           closed=closed)
    scene.add_shapes([path], stroke_color=([0.0, 0.0, 0.0], False) if cfg.stroke_w > 0 else None,
                     fill_color=([clr, clr, clr], True), split_primitives=False)
    # Currently color assignment wont work if we split

# Create logits for palette
num_colors = len(startup_paths)
num_colors += 1
#color_logits = torch.tensor(initial_logits, device=device)+ torch.randn((num_colors, cfg.K), device=device)*0.1
color_logits = torch.randn((num_colors, cfg.K), device=device)*0.5
color_logits.requires_grad = True
# with torch.no_grad():
#     color_logits.data.clamp_(0.0, 1.0)

#ref_size = np.max([geom.chord_length(P[:,:2]) for P in startup_paths])

Opt = torch.optim.Adam #Adam
Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.999)) #, eps=1e-6)

optimizers = [Opt(scene.get_points(), lr=cfg.lr_shape), #1*lrscale),
              Opt([color_logits], lr=cfg.lr_color),
              #Opt(scene.get_fill_colors(),  lr=cfg.lr_color),
              # Opt(scene.get_stroke_widths(), lr=0.5*lrscale)
              ]


schedulers = [util.step_cosine_lr_scheduler(opt, 0.0, cfg.lr_min_scale, cfg.num_opt_steps) for opt in optimizers]

##############################################
# Losses


losses = util.MultiLoss(verbose=verbose)
losses.add('mse',
           image_losses.MultiscaleMSELoss(rgb=False), cfg.mse_w)
if cfg.b_spline:
    losses.add('deriv',
               spline_losses.make_deriv_loss(cfg.deriv, 10), cfg.smoothing_w)

if True: #cfg.b_spline:
    losses.add('repulsion',
               spline_losses.make_repulsion_loss(cfg.repulsion_subd, False,
                                                 single=True,
                                                 signed=True, dist=cfg.repulsion_d), cfg.repulsion_w)

losses.add('bbox',
           spline_losses.make_bbox_loss(geom.make_rect(0, 0, w, h)), 1.0)


# style_loss = patch_loss = image_losses.CLIPPatchLoss(rgb=True, image_prompts=[style_img],
#                                                      model=cfg.clip_model,
#                                                      n_cuts=16)
style_loss = image_losses.CLIPPatchLoss(rgb=False, image_prompts=[style_img],
                                        min_size=200, #28,
                                        n_cuts=64,
                                        distortion_scale=0.3,
                                        blur_sigma=0,
                                        model='CLIPAG', use_negative=False) #clipag=cfg.use_clipag)


losses.add('style', style_loss, cfg.style_w)

if cfg.clipasso:
    clip_loss = image_losses.CLIPVisualLoss(rgb=cfg.use_color,
                                            clip_model=cfg.clip_model,
                                            distortion_scale=0.5,
                                            crop_scale=(0.8, 0.9), #0.6, 0.7), # 0.9,
                                            layer_weights=cfg.clip_layer_weights,
                                            semantic_w=cfg.clip_semantic_w)
    losses.add('clip',
               clip_loss, cfg.clip_w)
if cfg.sds:
    sds = sd.SDSLoss("A painting",
                     augment=4,
                     rgb=True,
                     controlnet="lllyasviel/sd-controlnet-canny",
                     #controlnet="lllyasviel/sd-controlnet-scribble",
                     seed=cfg.seed, #777, #999,
                    t_range=[cfg.t_min, cfg.t_max],
                    guidance_scale=7.5,
                    conditioning_scale=cfg.cond_scale,
                    num_hifa_denoise_steps=4,
                    time_schedule='ism', #'dtc', #'dtc', #'pow', #dtc', #linear', #'dtc', #'random', #'dreamtime', #'dtc', #'dreamtime', #'pow',
                    grad_method=cfg.grad_method, #'hifa', #ism', #csd', #ism', #'ism', #'ism', #'sds',
                     guess_mode=False)
    def sds_loss(im, step):
        return sds(im, cond_img, step, cfg.num_opt_steps,
                grad_scale=0.01 if sds.grad_method != 'hifa' else 1.0
                )
    losses.add('sds',
               sds_loss, cfg.sds_w)

##############################################
# Begin visualization and optimize

plut.set_theme()

fig = plt.figure(figsize=(8,7))
gs = GridSpec(3, 3, height_ratios=[1.0, 0.25, 0.25])
gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :])

first_frame = None
time_count = 0

def frame(step):
    global first_frame, time_count

    perf_t = time.perf_counter()

    for opt in optimizers:
       opt.zero_grad()

    #tau = tau_start + (tau_end - tau_start) * np.clip(step/(cfg.num_opt_steps*0.25), 0.0, 1.0)
    #decay_rate = 1e-3

    # Setup colors
    tau_start = cfg.tau_start #5.0 #1.0
    tau_end = 0.1
    decay_rate = np.log(tau_end / tau_start) / cfg.num_opt_steps


    tau = max(tau_end, tau_start * np.exp(decay_rate * step))

    #print('tau', tau)
    #cfg.gumbel_hard = False
    if cfg.gumbel_hard:
        soft_assign = F.gumbel_softmax(color_logits, tau=tau, hard=True)
    else:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(color_logits)))
        soft_assign = F.softmax((color_logits + cfg.gumbel_scale*gumbel_noise) / tau, dim=-1)


    # cfg.gumbel_hard = False
    # if cfg.gumbel_hard:
    #     soft_assign = F.gumbel_softmax(color_logits, tau=tau, hard=True)
    # else:
    #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(color_logits)))
    #     soft_assign = F.softmax((color_logits + cfg.gumbel_scale*gumbel_noise) / tau, dim=-1)

    #soft_assign = F.softmax(color_logits / tau, dim=-1)
    colors = soft_assign @ cfg.palette
    #pdb.set_trace()
    if cfg.gumbel_hard:
        quantized_hard = colors #.detach().cpu().numpy()
        indices = soft_assign
    else:
        with torch.no_grad():
            indices = torch.argmax(color_logits, dim=-1)
            quantized_hard = cfg.palette[indices]

    if cfg.lab:
        colors = lab2rgb(colors)
        quantized_hard = lab2rgb(quantized_hard)

    quantized_hard = quantized_hard.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()

    #print(colors.shape, len(scene.shape_groups))
    for i, (group, color) in enumerate(zip(scene.shape_groups, colors[:-1])):
        #color = color_logits[i,:1]
        group.fill_color = color #torch.tensor([1.0]).to(device)
        scene.groups[i]._fill_opt = color
        if cfg.stroke_w > 0.0:
            group.stroke_color = color*cfg.stroke_darkness #torch.tensor([1.0]).to(device)
            scene.groups[i]._stroke_opt = color*cfg.stroke_darkness

    # with torch.no_grad():
    #     indices = torch.argmax(color_logits, dim=-1)
    #     quantized_hard = cfg.palette[indices]
    #     if cfg.lab:
    #         quantized_hard = lab2rgb(quantized_hard)
    #     quantized_hard = quantized_hard.detach().cpu().numpy()

    background_image = torch.ones((h, w, 3), device=device)*colors[-1]

    # Rasterize
    try:
        with util.perf_timer('render', verbose=verbose):
            im = scene.render(background_image, num_samples=2)[:,:,:3].to(device)
    except RuntimeError as e:
        print(e)
        raise RuntimeError('error in render')
        #pdb.set_trace()

    #print([np.min(geom.chord_lengths(p.points.detach().cpu().numpy(), closed=True)) for p in scene.primitives])

    im_no_gamma = im
    im = im # ** gamma

    # Losses (see weights above)
    loss = losses(
        mse=(im, input_img, cfg.mse_mul),
        clip=(im, input_img), #target_tensor),
        clipag=(im,),
        sds=(im, step),
        style=(im,),
        deriv=(scene.shapes,),
        overlap=(im_no_gamma, scene.shapes,),
        repulsion=(scene.shapes,),
        bbox=(scene.shapes,),
    )

    mean_usage = soft_assign.mean(dim=0)
    print('Mean sum', mean_usage.sum().item())
    uniform = torch.full_like(mean_usage, 1.0 / cfg.K)

    # Match usage to uniform target
    palette_usage_loss = 250*F.mse_loss(mean_usage, uniform)
    print('palette_usage_loss', palette_usage_loss)
    loss += palette_usage_loss

    # Add reg for color logits
#     entropy = -(soft_assign * torch.log(soft_assign + 1e-8)).sum(dim=-1).mean()
#     lambda_entropy = 0.1 #100
#     reg = lambda_entropy * entropy  # try lambda_entropy = 0.01 or 0.1
#     #print('reg', reg.item())
#     loss += reg
# #
    with util.perf_timer('Opt step', verbose=verbose):
        loss.backward()
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

    # Constrain
    with torch.no_grad():
        pass
        # for path in paths:
        #     path.param('stroke_width').data.clamp_(minw, maxw)
        #     #path.param('stroke_width').data[0] = minw
        #     #path.param('stroke_width').data[-1] = minw
        # color_logits.data.clamp_(0.0, 1.0)
        #for i, clr in enumerate(scene.get_fill_colors()):
        #    clr.data.clamp_(0.0, 1.0)

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed
    lrs = 'lr %.3f'%(optimizers[0].param_groups[0]['lr'])

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

    # for group in scene.shape_groups:
    #     for shape in group.shapes:
    #         X = shape.samples(shape.num_points()*4).detach().cpu().numpy()
    #         plut.fill(X, np.ones(3)*group.fill_color.detach().cpu().numpy())
    # plut.setup()
    #     Q = path.param('points').detach().cpu().numpy()
    #     P = Q[::multiplicity]
    #     plut.stroke(P, 'r', lw=1.3, label='Keypoints' if i==0 else '', alpha=0.5)
    #     plut.stroke(Q, 'c', lw=0.5, label='Control Polygon' if i==0 else '', alpha=0.5)
    plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)


    plt.subplot(gs[0,1])
    if losses.has_loss('sds'):
        plt.title('Step %d, t %d, %s' %(step, int(sds.t_saved), lrs))
    else:
        plt.title('Step %d, tau %.2f lr %s'%(step, tau, lrs))
    #plt.title('Step %d'%step)
    plt.imshow((im*255).astype(np.uint8)) #, cmap='gray', vmin=0, vmax=1)

    plt.subplot(gs[0,2])
    bg = np.ones((h, w, 3))*quantized_hard[-1]
    plt.imshow(bg, vmin=0, vmax=1)


    resolve = False
    if cfg.resolve_self_ins_every > 0:
        resolve = step%cfg.resolve_self_ins_every == cfg.resolve_self_ins_every-1


    z = 0

    must_resolve = []
    print('Resolve: ', resolve)
    with torch.no_grad():
        for i, group in enumerate(scene.shape_groups):
            clr = np.ones(3)*quantized_hard[i] #  group.fill_color.detach().cpu().numpy()
            stroke_clr = clr*cfg.stroke_darkness
            #clr = np.ones(3)*group.fill_color.detach().cpu().numpy()
            #clr = np.ones(3)*colors[i].detach().cpu().numpy()
            for path in group.shapes:
                points = path.param('points')
                Q = points.detach().cpu().numpy()
                X = path.samples(len(Q)*10).detach().cpu().numpy()
                if closed: # and not overlap:
                    fill = [0.0, 0.0, 0.0, 0.1] #scene.shape_groups[i].fill_color.detach().cpu().numpy()
                    a = fill[-1]
                plut.fill(X[:,:2], clr[:3], zorder=z)
                if cfg.stroke_w > 0:
                    plut.stroke(X[:,:2], stroke_clr, lw=cfg.stroke_w*0.5, zorder=z+1)
                z += 3
                if resolve:
                    S1 = clipper.intersection(X, X, clip_type='evenodd')
                    S2 = clipper.union(X, X, clip_type='evenodd')
                    if len(S1) > 1 or len(S2) > 1:
                        must_resolve.append(path)
                        plut.stroke(S1, 'm', closed=True, zorder=100000)
                        print("Removing self intersections")
                    else:
                        pass #plut.stroke(S, 'g', closed=True, lw=1.5, zorder=z+2) #, zorder=100000)



    if must_resolve:
        #optimizers[0].zero_grad(set_to_none=True)
        with torch.no_grad():
            for path in must_resolve:
                points = path.param('points')
                Q = points.detach().cpu().numpy()
                S = clipper.simplify_polygon(Q)
                #S = clipper.intersection(Q, Q, clip_type='evenodd')
                Q = S[np.argmax([abs(geom.chord_length(P)) for P in S])]
                plut.stroke(Q, 'c', closed=True, zorder=100000)
                #print('cl', geom.chord_length(Q), 'mincl', np.min(geom.chord_lengths(Q)))
                #Q = geom.cleanup_contour(Q, eps=1e-3, closed=True)
                new_points = torch.from_numpy(Q).to(torch.float32).to(device)
                #path.params['points'].data = torch.from_numpy(Q).to(torch.float32).to(device)
                new_points.requires_grad = True
                path.params['points'] = new_points
                points.requires_grad = False
                path.setup()
                path.refresh()

                # # Replace in point opt
                opt = optimizers[0]
                for param_group in opt.param_groups:
                    for j, param in enumerate(param_group['params']):
                        if param is points:
                            param_group['params'][j] = new_points
                            #pdb.set_trace()
            # old_opt = optimizers[0]
            # old_lr = old_opt.param_groups[0]['lr']
            # optimizers[0] = Opt(scene.get_points(), lr=old_lr)
            # del old_opt
            #schedulers[0].optimizer = optimizers[0]
        #optimizers[0] = Opt(scene.get_points(), lr=cfg.lr_shape)
        #del old_opt
        #torch.cuda.empty_cache()

    #must_resolve = []



    if losses.has_loss('curv'):
        plut.fill_circle([cfg.target_radius, cfg.target_radius], cfg.target_radius, 'c', zorder=100000)

    #plt.legend()
    plut.setup(box=geom.make_rect(0, 0, w, h))

    plt.subplot(gs_sub[0])
    plt.imshow(density_map)
    plt.subplot(gs_sub[1])
    plt.imshow(style_img)
    plt.subplot(gs_sub[2])
    for i, clr in enumerate(cfg.palette):
        plut.fill_rect(geom.make_rect(i, 0, 1, 1), clr.detach().cpu().numpy())
    plut.setup()
    plt.subplot(gs[2,:])
    plt.title('Loss')
    for i, (key, kloss) in enumerate(losses.losses.items()):
        if key=='total' or not losses.has_loss(key):
            # There is a bug in 'total'
            continue
        plt.plot(kloss, label='%s:%.4f'%(key, kloss[-1]))
    plt.legend()

    if saver.valid:
        save_every = 10
        if step%save_every == save_every-1:
            saver.clear_collected_paths()
            scene.save_json(saver.with_ext('.json'), background_color=quantized_hard[-1],
                            colors=quantized_hard[:-1],
                            bg_index=indices[-1],
                            indices=indices[:-1])
            cfg.save_yaml(saver.with_ext('.yaml'))
            plut.figure_image().save(saver.with_ext('.png'))
            saver.log_image('output', plt.gcf())
            saver.copy_file()


#frame(0)
plut.show_animation(fig, frame, cfg.num_opt_steps, filename=saver.with_ext('.mp4'), headless=cfg.headless)
saver.finish()
