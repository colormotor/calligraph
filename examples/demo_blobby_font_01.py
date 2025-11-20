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
                        svg,
                        sd,
                        config,
                        util,
                        diffvg_utils,
                        imaging,
                        stroke_init,
                        spline_losses,
                        image_losses,
                        )

import time
device = config.device
dtype = torch.float32

def params():
    output_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/outputs/'
    save = True

    ablation = ''

    text = 'R'
    suffix = text

    sides = 3 
    ds = 15
    offset = 0 

    image_alpha = 1 #0.9
    
    degree =  5 #5
    deriv = 3
    multiplicity = 1
    b_spline = 1
    catmull = 0
    pspline = 0
    ref_size_factor = 1.0

    lr_shape = 1.0 
    num_opt_steps = 200 
    
    # OCR: no cls, av pool
    # BLIP: only cls
    use_ocr = True
    force_cls = None 
    pool_no_cls = True

    if use_ocr:
        semantic_w = 1000 
        if force_cls is True:
            semantic_w = 5000
    else:
        semantic_w = 180 

    smoothing_w = 5000 #3000 #1000 #700 # 200 #100 #100 # 10.0 # 1.0 #10.0 # 0.0 #0.0 #1 #0.1

    stroke_w = 1.0 #5 #0.5

    repulsion_subd = 10 #25 # 15 #0 #15 #15 #10 #5
    # if not b_spline:
    #     repulsion_subd = repulsion_subd//2
    repulsion_w = 10000 #5000 #0#  5000 #0.1 #10 #2000 #1000 #15000
    resolve_self_ins_every = 0 # 5 if repulsion_w else 0 #5 # 5 #1
    repulsion_d = 10 #20

    mse_w = 35 #100 #10 #0.5 #100.0
    mse_mul = 2 # Factor multiplying each mse blur level (> 1 emph low freq)

    seed = 1233

    return locals()


cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = '___'

cfg.ablation = cfg.ablation.lower()
if 'r' in cfg.ablation:
    cfg.repulsion_w = 0.0
    cfg.resolve_self_ins_every = 0
if 's' in cfg.ablation:
    cfg.smoothing_w = 0.0
if 'o' in cfg.ablation:
    cfg.semantic_w = 0.0
if 'b' in cfg.ablation:
    cfg.b_spline = 0
if cfg.ablation:
    cfg.suffix += '_abl-' + cfg.ablation
saver = util.SaveHelper(__file__, output_path, 
                        suffix=cfg.suffix,
                        cfg=cfg)


sz = 400 #256 # 512
box = geom.make_rect(0, 0, sz, sz)

# Create target text image and outline
text_img, letter_shape = plut.font_to_image(cfg.text, (sz, sz), return_outline=True, padding=0)
letter_shape = geom.fix_shape_winding(letter_shape)

# Target image. This can be any silhouette but we generate a regular polygon here
plut.figure_pixels(sz, sz)
P = geom.shapes.regular_polygon([sz/2, sz/2], sz*0.4, n=cfg.sides)
plut.fill(P, 'k')
plut.setup(box=box)
input_img = plut.figure_image(close=True).convert('L')

# Move outline to approximately cover target area
def fit_shape_to_polygon(S, P):
    Q = np.vstack(S)
    pq = np.mean(Q, axis=0)
    norms = np.linalg.norm(Q-pq, axis=1)
    rq = np.max(norms)
    pp = np.mean(P, axis=0)
    rp = np.min([geom.point_segment_distance(pp, a, b) for a, b in zip(P, P[1:])])
    return geom.tsm(geom.trans_2d(pp)@geom.scaling_2d(rp/rq)@geom.trans_2d(-pq),
                    S), rp

letter_shape, r = fit_shape_to_polygon(letter_shape, P)

cfg.ds = r/5
cfg.repulsion_d = r/5

plut.figure_pixels(sz, sz)
plut.fill(letter_shape, 'k')
plut.setup(box=box)
text_img = plut.figure_image(close=True).convert('L')

letter_shape = [geom.uniform_sample(P, cfg.ds, closed=True) for P in letter_shape]
outlines = letter_shape


im = np.array(input_img)/255
ctrs = clipper.multi_union([imaging.find_contours(1-im)]*2)
if cfg.offset:
    area = clipper.offset(clipper.offset(ctrs, -cfg.offset), cfg.offset, join_type='round')
else:
    area = ctrs
rast = imaging.ShapeRasterizer(box, sz)
rast.fill_shape(area)
input_img = rast.Image(invert=True)
mse_img = input_img


# plut.figure_pixels(sz, sz)
# plut.fill(area, 'k')
# plut.setup()
# input_img = plut.figure_image(True).convert('L')

complement = clipper.difference(ctrs, area)


img = np.array(input_img)/255
h, w = img.shape

mse_img = np.array(mse_img)/255
mse_img = 1-(1-mse_img)*cfg.image_alpha

outlines = [np.array(P).copy() for P in outlines]

##############################################
# Settings
verbose = False

overlap = False
ref_size = w/cfg.ref_size_factor
offset_variance = [ref_size, ref_size]
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
#outlines = [geom.uniform_sample(geom.cleanup_contour(X, closed=True), cfg.ds, closed=True) for X in outlines]
#outlines = geom.fix_shape_winding(outlines)


scene = diffvg_utils.Scene()

closed = True
paths = []
for P in outlines:
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(P,
                degree=cfg.degree,
                stroke_width=(cfg.stroke_w, False),
                pspline=cfg.pspline,
                closed=closed)
    else:
        if not cfg.catmull:
            Q = bezier.cubic_bspline_to_bezier_chain(P, periodic=closed)
            if closed:
                Q = Q[:-1]
            path = diffvg_utils.Path(Q[:,:2],
                                     degree=3,
                                     stroke_width=(cfg.stroke_w, False),
                                     closed=closed)
        else:
            path = diffvg_utils.CardinalSpline(P,
                                           stroke_width=(cfg.stroke_w, False),
                                           closed=closed)
    paths.append(path)

scene.add_shapes(paths, stroke_color=([0.0, 0.0, 0.0], False) if cfg.stroke_w > 0 else None,
                     fill_color=([0.0], False), split_primitives=False)
    # Currently color assignment wont work if we split


Opt = torch.optim.Adam #Adam
Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.9), eps=1e-6)
Opt = lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.999)) #, eps=1e-6)

optimizers = [Opt(scene.get_points(), lr=cfg.lr_shape), #1*lrscale),
              ]


schedulers = [util.step_cosine_lr_scheduler(opt, 0.0, 0.6, cfg.num_opt_steps) for opt in optimizers]
#schedulers = []

##############################################
# Losses

losses = util.MultiLoss(verbose=verbose)

losses.add('mse',
           image_losses.MultiscaleMSELoss(rgb=False), cfg.mse_w)


if cfg.b_spline: # b_spline:
    losses.add('deriv',
               spline_losses.make_deriv_loss(cfg.deriv, w), cfg.smoothing_w)

if True: #cfg.b_spline:
    losses.add('repulsion',
               spline_losses.make_repulsion_loss(cfg.repulsion_subd, False,
                                                 single=False,
                                                 signed=True, dist=cfg.repulsion_d), cfg.repulsion_w)

losses.add('bbox',
           spline_losses.make_bbox_loss(geom.make_rect(0, 0, w, h)), 1.0)


from transformers import BlipProcessor, BlipModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


if cfg.use_ocr:
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    cls = True
else:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    cls = True

if cfg.force_cls is not None:
    cls = cfg.force_cls

vis_loss = image_losses.VisionEncoderLoss(model, processor,
                                              layer_weights=[
                                                             ],
                                            cls=cls,
                                            pool_no_cls=cfg.pool_no_cls,
                                              metric='L1', #'cosine', #l2',
                                             rgb=False)

#
def semantic_loss(im, other):
    #return clip_semantic_loss(im)*0.01
    return vis_loss(im, other)

losses.add('semantic',
               semantic_loss, cfg.semantic_w) #cfg.vis_w*0.001)

##############################################
# Begin visualization and optimize

plut.set_theme()

fig = plt.figure(figsize=(8,7))
gs = GridSpec(2, 3, height_ratios=[1.0, 0.4])
#gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :])

first_frame = None
time_count = 0

def frame(step):
    global first_frame, time_count

    perf_t = time.perf_counter()

    for opt in optimizers:
       opt.zero_grad()

    background_image = torch.ones((h, w), device=device)

    # Rasterize
    try:
        with util.perf_timer('render', verbose=verbose):
            im = scene.render(background_image, num_samples=2)[:,:,0].to(device)
    except RuntimeError as e:
        print(e)
        pdb.set_trace()

    #print([np.min(geom.chord_lengths(p.points.detach().cpu().numpy(), closed=True)) for p in scene.primitives])

    im_no_gamma = im
    im = im # ** gamma

    mid_step = cfg.num_opt_steps/2
    # if step > mid_step:
    #     amt = np.clip(1.0-(step-mid_step)/mid_step, 0.0, 1.0)
    #     print('annealing semantic', amt)
    #     losses.anneal_weights(deriv=amt)
    # Losses (see weights above)
    loss = losses(
        mse=(im, mse_img, cfg.mse_mul,),
        clip=(im, input_img,), #target_tensor),
        clipag=(im,),
        semantic=(im, text_img), #input_img), #target_tensor),
        sds=(im, step),
        curv=(scene.shapes,),
        flow=(scene.shapes,),
        orient=(scene.shapes,),
        deriv=(scene.shapes,),
        repulsion=(scene.shapes,),
        bbox=(scene.shapes,),
    )

    # Add reg for color logits
    # entropy = -(soft_assign * torch.log(soft_assign + 1e-8)).sum(dim=-1).mean()
    # lambda_entropy = 0.1 #100
    # reg = lambda_entropy * entropy  # try lambda_entropy = 0.01 or 0.1
    #print('reg', reg.item())
    #loss += reg
#
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
    plt.imshow((np.array(text_img)/255)*amt + mse_img*(1-amt) , cmap='gray')

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
        plt.title('Step %d, %s'%(step, lrs))
    #plt.title('Step %d'%step)
    plt.imshow(im * amt + mse_img*(1-amt), cmap='gray', vmin=0, vmax=1)

    # plt.subplot(gs[0,2])
    # bg = np.ones((h, w, 3)) #*quantized_hard[-1]
    # plt.imshow(bg, vmin=0, vmax=1)


    resolve = False
    if cfg.resolve_self_ins_every > 0:
        resolve = step%cfg.resolve_self_ins_every == cfg.resolve_self_ins_every-1


    z = 0

    plt.subplot(gs[0,2])
    
    with torch.no_grad():
        S = []
        for i, group in enumerate(scene.shape_groups):
            #clr = np.ones(3)*quantized_hard[i] #  group.fill_color.detach().cpu().numpy()
            clr = np.ones(3)*group.fill_color.detach().cpu().numpy()
            #clr = np.ones(3)*colors[i].detach().cpu().numpy()
            for path in group.shapes:
                points = path.param('points')
                Q = points.detach().cpu().numpy()
                X = path.samples(len(Q)*10).detach().cpu().numpy()
                S.append(X)
    

        plut.fill(S, 'k')
        plut.stroke(S, 'k', lw=cfg.stroke_w*0.5, closed=True)
        if cfg.offset > 0:
            S = clipper.union(S, S)
            S = clipper.offset(clipper.offset(S, cfg.offset, join_type='round'), cfg.offset) #, -cfg.offset)
            compl = clipper.difference(ctrs, S)
            plut.fill(compl, 'k')
            plut.fill(complement, 'k')

    if losses.has_loss('curv'):
        plut.fill_circle([cfg.target_radius, cfg.target_radius], cfg.target_radius, 'c')

    plut.setup(box=geom.make_rect(0, 0, w, h))

    # plt.subplot(gs_sub[0])
    # plt.imshow(density_map)
    # plt.subplot(gs_sub[1])
    # plt.imshow(style_img)
    # plt.subplot(gs_sub[2])
    # for i, clr in enumerate(cfg.palette):
    #     plut.fill_rect(geom.make_rect(i, 0, 1, 1), clr.detach().cpu().numpy())
    # plut.setup()
    plt.subplot(gs[1,:])
    plt.title('Loss')
    for i, (key, kloss) in enumerate(losses.losses.items()):
        if key=='total' or not losses.has_loss(key):
            # There is a bug in 'total'
            continue
        plt.plot(kloss, label='%s:%.4f'%(key, kloss[-1]))
    plt.legend()

    if saver.valid:
        save_every = 10
        store_every = 50
        if step%save_every == save_every-1:
            saver.clear_collected_paths()
            scene.save_json(saver.with_ext('.json'), complement=complement, area=area, ctrs=ctrs)
            cfg.save_yaml(saver.with_ext('.yaml'))
            plut.figure_image().save(saver.with_ext('.png'))
            saver.log_image('output', plt.gcf())
            saver.copy_file()

        if step%store_every == store_every-1:
            ind = step//store_every
            scene.save_json(saver.with_ext('.json', suffix=f'sub{ind}'), complement=complement, area=area, ctrs=ctrs)
            plut.figure_image().save(saver.with_ext('.png', suffix=f'sub{ind}'))


#frame(0)
plut.show_animation(fig, frame, cfg.num_opt_steps, filename=saver.with_ext('.mp4'), headless=cfg.headless)
saver.finish()
