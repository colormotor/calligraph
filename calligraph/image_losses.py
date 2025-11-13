#!/usr/bin/env python3
"""
 )   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

Loss functions for image comparison

TODO what?? Adapted from:
- https://github.com/jiupinjia/stylized-neural-painting
"""

import importlib
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as Fv
import torch.nn.functional as F
from collections import OrderedDict

from . import config, util, geom

device = config.device

cfg = lambda: None
cfg.batch_size = 1  # Hacky, fixme
cfg.clip_models = {}


def to_batch(x, rgb):
    if isinstance(x, Image.Image):
        if not rgb:
            x = x.convert("L")
        x = torch.tensor(np.array(x) / 255, device=device)
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    if rgb:
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        x = x.permute((3, 2, 0, 1))  # to NCHW
    else:
        if len(x.shape) > 2:
            x = torch.mean(x, axis=-1)
        if len(x.shape) == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        x = x.repeat(1, 3, 1, 1)
    x = x.repeat(cfg.batch_size, 1, 1, 1)
    return x


def make_total_variation_loss(rgb=True):
    def lossfn(image):
        image = to_batch(image, rgb=rgb)
        tv_loss = torch.sum(
            torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        ) + torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return tv_loss / (image.shape[-1] * image.shape[-2])

    return lossfn


class MSELoss(torch.nn.Module):
    """Multiscale MSE loss for images, adapted from PyDiffvg examples"""

    def __init__(self, rgb=True, blur=0, *args, **kwargs):
        super(MSELoss, self).__init__()
        self.rgb = rgb
        self.blur = blur

    def forward(self, im, target, *args):
        im = to_batch(im, self.rgb)
        target = to_batch(target, self.rgb).to(im.dtype)
        if self.blur > 0:
            sigma = self.blur
            k = int(np.ceil(4 * sigma) + 1)
            im = Fv.gaussian_blur(im, k, sigma)
            target = Fv.gaussian_blur(target, k, sigma)
        self.blur_target = target[0, 0, :, :]
        self.blur_im = im[0, 0, :, :]
        bs, c, h, w = im.shape
        return torch.nn.functional.mse_loss(im, target)


class MultiscaleMSELoss(torch.nn.Module):
    """Multiscale MSE loss for images, adapted from PyDiffvg examples"""

    def __init__(self, sigma=1, rgb=True, debug=False):
        super(MultiscaleMSELoss, self).__init__()
        self.rgb = rgb
        self.blur = transforms.GaussianBlur(
            kernel_size=int(np.ceil(4 * sigma)) + 1, sigma=(sigma, sigma)
        )
        self.debug = debug

    def forward(self, im, target, mult=1, scale_factor=0.5, num_levels=None):
        im = to_batch(im, self.rgb)
        target = to_batch(target, self.rgb).to(im.dtype)
        # print(im.dtype, target.dtype)
        bs, c, h, w = im.shape

        if num_levels is None:
            num_levels = max(int(np.ceil(np.log2(h))) - 2, 1)

        sz = im.shape[-1]
        losses = []
        w = 1.0
        ims = []
        targets = []
        wsum = 0
        for lvl in range(num_levels):
            loss = torch.nn.functional.mse_loss(im, target)
            losses.append(loss * w)
            wsum += w
            if self.debug:
                # ims.append(im)
                ims.append(
                    torch.nn.functional.interpolate(
                        im,
                        scale_factor=sz / im.shape[-1],
                        align_corners=True,
                        mode="bicubic",
                    )
                    * w
                )
                targets.append(
                    torch.nn.functional.interpolate(
                        target,
                        scale_factor=sz / target.shape[-1],
                        align_corners=True,
                        mode="bicubic",
                    )
                    * w
                )
            w = w * mult

            im = torch.nn.functional.interpolate(
                self.blur(im), scale_factor=scale_factor, mode="nearest"
            )
            target = torch.nn.functional.interpolate(
                self.blur(target), scale_factor=scale_factor, mode="nearest"
            )

        if self.debug:
            self.blur_im = (sum(ims) / wsum)[0, 0, :, :]
            self.blur_target = (sum(targets) / wsum)[0, 0, :, :]

        losses = torch.stack(losses)
        return losses.sum()


def tv_loss(im, rgb=True):
    im = to_batch(im, rgb)
    w_variance = torch.mean(torch.pow(im[:, :, :, :-1] - im[:, :, :, 1:], 2))
    h_variance = torch.mean(torch.pow(im[:, :, :-1, :] - im[:, :, 1:, :], 2))
    return (h_variance + w_variance) / 2.0


class CLIPPatchLoss(torch.nn.Module):
    def __init__(
        self,
        text_prompts=[],
        image_prompts=[],
        negative_prompts=["A badly drawn sketch.", "Many ugly, messy drawings."],
        use_negative=True,
        model="ViT-B-32",
        crop_scale=(0.6, 0.9),
        distortion_scale=0.0,
        thresh=0.5,
        blur_sigma=0.0,
        min_size=None,
        cut_scale=0.25,
        num_batches=1,
        n_cuts=16,
        rgb=True,
        clipag=False,
    ):
        super().__init__()  # super(CLIPPatchLoss, self).__init__()
        from torchvision import transforms
        import open_clip

        self.rgb = rgb
        if clipag:
            model = "CLIPAG"

        self.clip_model, _, tokenizer, self.clip_model_input_size = load_clip_model(
            model
        )
        # self.clip_model_input_size = 224
        if not use_negative:
            negative_prompts = []
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    size=self.clip_model_input_size, max_size=None, antialias=None
                ),
                # transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
                transforms.CenterCrop(
                    size=(self.clip_model_input_size, self.clip_model_input_size)
                ),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.clip_model.to(device)

        self.clip_model.eval()

        self.target_embeds = []
        self.negative_embeds = []
        image_prompts = [to_batch(img, rgb).to(torch.float32) for img in image_prompts]
        with torch.no_grad():
            for text_prompt in text_prompts:
                tokenized_text = tokenizer([text_prompt]).to(device)
                self.target_embeds.append(self.clip_model.encode_text(tokenized_text))
            for image_prompt in image_prompts:
                image_embed = self.clip_model.encode_image(
                    self.preprocess(image_prompt)
                )
                self.target_embeds.append(image_embed)
            for text_prompt in negative_prompts:
                tokenized_text = tokenizer([text_prompt]).to(device)
                self.negative_embeds.append(self.clip_model.encode_text(tokenized_text))

        self.num_positive = len(self.target_embeds)
        self.num_negative = len(self.negative_embeds)

        self.target_embeds = torch.cat(self.target_embeds)
        if self.negative_embeds:
            self.negative_embeds = self.negative_embeds
        else:
            self.negative_embeds = None
        self.n_cuts = n_cuts
        self.num_batches = num_batches
        self.cut_scale = cut_scale
        self.thresh = thresh

        augment_list = []
        if distortion_scale > 0:
            augment_list.append(
                transforms.RandomPerspective(
                    fill=1, p=1.0, distortion_scale=distortion_scale
                )  # 0.5)
            )
        # augment_list.append(
        #         transforms.RandomResizedCrop(self.clip_model_input_size, scale=crop_scale, ratio=(1.0, 1.0))
        #     )
        if blur_sigma > 0.0:
            augment_list.append(
                transforms.GaussianBlur(
                    kernel_size=5, sigma=(blur_sigma * 0.01, blur_sigma)
                )  # Example: 5x5 kernel
            )
        # augment_list.append(self.clip_norm_)  # CLIP Normalize
        # compose augmentations
        self.augment_compose = transforms.Compose(augment_list)
        if min_size is None:
            self.min_size = self.clip_model_input_size
        else:
            self.min_size = min_size

    def forward(self, input, *args):
        input = to_batch(input, self.rgb)

        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.min_size)

        cuts_per_batch = self.n_cuts // self.num_batches
        remaining_cuts = self.n_cuts % self.num_batches
        loss = 0
        cutout_count = 0

        for batch_idx in range(self.num_batches):
            n_cuts_this_batch = cuts_per_batch + (
                1 if batch_idx < remaining_cuts else 0
            )
            cutouts = []

            for _ in range(n_cuts_this_batch):
                size = int(
                    torch.rand([]) * self.cut_scale * (max_size - min_size) + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutout = torch.nn.functional.adaptive_avg_pool2d(
                    cutout, self.clip_model_input_size
                )
                cutout = self.augment_compose(cutout)
                cutouts.append(cutout)

            cutouts_batch = torch.cat(cutouts, dim=0)

            if batch_idx == 0:
                self.test_cutout = cutouts_batch[0].detach().cpu().numpy()[0, :, :]

            input_embeds = self.clip_model.encode_image(self.preprocess(cutouts_batch))

            for n in range(n_cuts_this_batch):
                patch_loss = torch.cosine_similarity(
                    self.target_embeds, input_embeds[n : n + 1], dim=1
                )
                if self.thresh > 0 and patch_loss < self.thresh:
                    patch_loss = 0
                loss -= patch_loss

                if self.negative_embeds is not None:
                    div = 1 / self.num_negative
                    for feat in self.negative_embeds:
                        loss += (
                            torch.cosine_similarity(
                                feat, input_embeds[n : n + 1], dim=1
                            )
                            * div
                        )

                cutout_count += 1

            del cutouts_batch, input_embeds
            torch.cuda.empty_cache()  # Optional, for GPU

        return loss[0] / cutout_count


class LPIPS(torch.nn.Module):
    """LPIPS loss, requires piq package"""

    def __init__(
        self, rgb=True, size=224, distortion_scale=0.5, crop_scale=(0.8, 1.0), **kwargs
    ):
        super(LPIPS, self).__init__()
        import piq

        self.rgb = rgb
        self.size = size
        self.LPIPS = piq.LPIPS(**kwargs)
        self.distortion_scale = distortion_scale
        self.crop_scale = crop_scale

    def forward(self, x, y):
        x = to_batch(x, self.rgb)
        y = to_batch(y, self.rgb).to(dtype=x.dtype)

        if self.size is not None:
            x = torch.nn.functional.interpolate(
                x, mode="bilinear", size=(self.size, self.size), align_corners=False
            )
            y = torch.nn.functional.interpolate(
                y, mode="bilinear", size=(self.size, self.size), align_corners=False
            )

        # init augmentations
        augment_list = []
        augment_list.append(
            transforms.RandomPerspective(
                fill=1, p=1.0, distortion_scale=self.distortion_scale
            )  # 0.5)
        )
        augment_list.append(
            transforms.RandomResizedCrop(
                self.size, scale=self.crop_scale, ratio=(1.0, 1.0)
            )
        )
        augment_compose = transforms.Compose(augment_list)
        # make augmentation pairs
        x_augs, y_augs = [x], [y]
        # repeat N times
        for n in range(4):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))

        x = torch.cat(x_augs, dim=0)
        y = torch.cat(y_augs, dim=0)

        return self.LPIPS(x, y)


class VGGPerceptualLoss(torch.nn.Module):
    """VGG perceptual loss
    TODO: Test RGB
    """

    def __init__(
        self, rgb=True, resize=True, distortion_scale=0.5, crop_scale=(0.8, 1.0)
    ):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize
        self.rgb = rgb
        self.distortion_scale = distortion_scale
        self.crop_scale = crop_scale

    def forward(self, input, target, ignore_color=False):
        # Assume 0-1 input
        input = to_batch(input, self.rgb)
        target = to_batch(target, self.rgb).to(dtype=input.dtype)

        self.mean = self.mean.type_as(input)
        self.std = self.std.type_as(input)
        if ignore_color:
            input = torch.mean(input, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)
        # if input.shape[1] != 3:
        #    input = input.repeat(1, 3, 1, 1)
        #    target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0

        x = input
        y = target

        # init augmentations
        augment_list = []
        if self.distortion_scale > 0.0:
            augment_list.append(
                transforms.RandomPerspective(
                    fill=1, p=1.0, distortion_scale=self.distortion_scale
                )  # 0.5)
            )
        augment_list.append(
            transforms.RandomResizedCrop(224, scale=self.crop_scale, ratio=(1.0, 1.0))
        )
        augment_compose = transforms.Compose(augment_list)
        # make augmentation pairs
        x_augs, y_augs = [x], [y]
        # repeat N times
        for n in range(4):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))

        x = torch.cat(x_augs, dim=0)
        y = torch.cat(y_augs, dim=0)

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def get_model_image_size(processor):
    """
    Safely extract image input size from a Hugging Face processor.
    Works with TrOCR, BLIP, CLIP, etc.
    """
    size = None

    # Try modern `image_processor` field
    if hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "size"
    ):
        size = processor.image_processor.size
    # Fall back to legacy `feature_extractor`
    elif hasattr(processor, "feature_extractor") and hasattr(
        processor.feature_extractor, "size"
    ):
        size = processor.feature_extractor.size

    if isinstance(size, dict):
        return tuple(size.values())  # {'height': 224, 'width': 224} → (224, 224)
    elif isinstance(size, int):
        return (size, size)
    elif isinstance(size, (list, tuple)):
        return tuple(size)
    else:
        raise ValueError("Cannot determine image size from processor.")


def parse_augments(size, augment_recs):
    augments = []
    augs = {
        "blur": lambda kwargs: T.GaussianBlur(**kwargs),
        "persp": lambda kwargs: T.RandomPerspective(fill=1, p=1, **kwargs),
        "rot": lambda kwargs: T.RandomRotation(**kwargs),
        "clr": lambda kwargs: T.ColorJitter(**kwargs),
        "aff": lambda kwargs: T.RandomAffine(**kwargs),
        "crop": lambda kwargs: T.RandomResizedCrop(size, **kwargs),
    }

    for kind, kwargs in augment_recs:
        if kind in augs:
            augments.append(augs[kind](kwargs))

    return T.Compose(augments)


class VisionEncoderLoss(torch.nn.Module):
    def __init__(
        self,
        model,
        processor,
        layer_weights=[],
        metric="cosine",  #'l2', #'cosine',
        rgb=False,
        question="",
        cls=False,
        augs=[],
        pool_no_cls=True,
        distortion_scale=0.0,
        crop_scale=(0.8, 1.0),
    ):
        super(VisionEncoderLoss, self).__init__()
        self.processor = processor
        self.model = model
        self.layer_weights = layer_weights
        self.rgb = rgb
        self.pool_no_cls = pool_no_cls

        # This disables all gradients
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.cls = cls

        if metric == "cosine":
            self.metric = lambda a, b: 1 - F.cosine_similarity(a, b).mean()
        elif metric == "L1":
            self.metric = lambda a, b: torch.abs(a - b).mean()
        elif metric == "wasserstein":
            self.metric = lambda a, b: wasserstein_loss(a, b)
        else:
            self.metric = F.mse_loss

        self.size = get_model_image_size(
            processor
        )  # tuple(processor.feature_extractor.size.values())
        print("Size", self.size)

        # Image Augmentation Transformation
        if augs:
            self.augment_trans = parse_augments(self.size, augs)
        else:
            augments = []

            if distortion_scale > 0:
                augments.append(
                    T.RandomPerspective(fill=1, p=1, distortion_scale=distortion_scale)
                )
            augments.append(T.RandomResizedCrop(self.size, scale=crop_scale))

            self.augment_trans = T.Compose(augments)

        self.resize_normalize = T.Compose(
            [
                T.Resize(self.size),
                # T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.question = question

    def forward(self, img_src, img_tgt, num_aug=0):
        hidden = True if self.layer_weights else False

        img_src = to_batch(img_src, self.rgb)
        img_tgt = to_batch(img_tgt, self.rgb).to(dtype=img_src.dtype)
        # img_src.requires_grad_()

        # augment (if specified)
        src_imgs = []  # self.resize_normalize(img_src)]
        tgt_imgs = []  # self.resize_normalize(img_tgt)]

        if num_aug > 0:
            for _ in range(num_aug):
                src_imgs.append(self.resize_normalize(self.augment_trans(img_src)))
                tgt_imgs.append(self.resize_normalize(self.augment_trans(img_tgt)))
        else:
            src_imgs.append(self.resize_normalize(img_src))
            tgt_imgs.append(self.resize_normalize(img_tgt))

        img_src = torch.cat(src_imgs, dim=0)
        img_tgt = torch.cat(tgt_imgs, dim=0)
        # img_src = self.resize_normalize(img_src)
        # img_tgt = self.resize_normalize(img_tgt)

        # return self.metric(img_src, img_tgt) #src_inputs, tgt_inputs)
        src_inputs = img_src
        tgt_inputs = img_tgt
        # src_inputs = self.processor(img_src, return_tensors='pt', do_resize=False).pixel_values.to(device)

        # tgt_inputs = self.processor(img_tgt, return_tensors='pt', do_resize=False).pixel_values.to(device)
        # src_inputs = torch.stack(self.processor.feature_extractor(img_src, return_tensors=None)['pixel_values']).to(device)
        # tgt_inputs = torch.stack(self.processor.feature_extractor(img_tgt, return_tensors=None)['pixel_values']).to(device)
        # return self.metric(src_inputs, tgt_inputs)

        try:
            encoder = self.model.encoder
        except AttributeError:
            encoder = self.model.vision_model

        src_states = encoder(
            src_inputs, output_hidden_states=hidden, output_attentions=hidden
        )
        tgt_states = encoder(
            tgt_inputs, output_hidden_states=hidden, output_attentions=hidden
        )
        assert not torch.isnan(src_states.last_hidden_state).any()
        # select = lambda state: state[:,1:,:].reshape(state.size(0), -1)
        if self.cls:
            select = lambda state: state[:, 0, :]
            # select = lambda state: F.normalize(state[:, 0, :], dim=-1)
            # select = lambda state: state[:,:,:].mean(dim=1)
        else:
            if self.pool_no_cls:
                select = lambda state: state[:, 1:, :].mean(dim=1)
            else:
                select = lambda state: state[:, :, :]  # .mean(dim=1)

        # select = lambda state: torch.cat([
        #     state[:, 0, :]*1,                    # CLS
        #     state[:, 1:, :].mean(dim=1)        # Patch mean
        # ], dim=-1) #*5000

        if hidden:
            loss = 0
            print("Num states", len(src_states.hidden_states))
            for i, w in self.layer_weights:
                src = select(src_states.hidden_states[i])
                tgt = select(tgt_states.hidden_states[i])
                loss += self.metric(src, tgt).mean() * w
            return (loss / len(self.layer_weights)) * 10000
        else:
            src_enc = select(src_states.last_hidden_state)
            tgt_enc = select(tgt_states.last_hidden_state)
            return self.metric(src_enc, tgt_enc).mean()


class CLIPSemanticLoss(torch.nn.Module):
    def __init__(
        self,
        prompt,
        neg_prompts=["A badly drawn sketch.", "Many ugly, messy drawings."],
        model="CLIPAG",
        use_negative=False,
        negative_weight=0.01,
        normalized=True,
        distortion_scale=0.5,
        crop_scale=(0.9, 0.9),
        rgb=True,
    ):
        super(CLIPSemanticLoss, self).__init__()
        self.model, self.preprocess, tokenizer, self.input_size = load_clip_model(model)
        # tokenizer = open_clip.get_tokenizer(model)
        self.model.eval()

        self.use_negative = use_negative
        self.negative_weight = negative_weight
        self.text_input = tokenizer(prompt).to(device)
        self.text_inputs_neg = [
            tokenizer(neg_prompt).to(device) for neg_prompt in neg_prompts
        ]

        # Calculate features
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_input)
            self.text_features_neg = [
                self.model.encode_text(neg) for neg in self.text_inputs_neg
            ]

        # Image Augmentation Transformation
        self.augment_trans = transforms.Compose(
            [
                transforms.RandomPerspective(
                    fill=1, p=1, distortion_scale=distortion_scale
                ),
                transforms.RandomResizedCrop(self.input_size, scale=crop_scale),
            ]
        )

        self.rgb = rgb

        if normalized:
            self.augment_trans = transforms.Compose(
                [
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomPerspective(
                        fill=1, p=1, distortion_scale=distortion_scale
                    ),
                    transforms.RandomResizedCrop(self.input_size, scale=crop_scale),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

    def forward(self, x, num_aug=4):

        x = to_batch(x, self.rgb)

        loss = 0
        # NUM_AUGS = 4
        img_augs = []
        if num_aug:
            for n in range(num_aug):
                img_augs.append(self.augment_trans(x))
        else:
            img_augs.append(x)
            num_aug = 1

        im_batch = torch.cat(img_augs)
        image_features = self.model.encode_image(im_batch)
        for n in range(num_aug):
            loss -= torch.cosine_similarity(
                self.text_features, image_features[n : n + 1], dim=1
            )
            if self.use_negative and self.text_features_neg:
                div = (1.0 / len(self.text_features_neg)) * self.negative_weight
                for feat in self.text_features_neg:
                    loss += (
                        torch.cosine_similarity(feat, image_features[n : n + 1], dim=1)
                        * div
                    )
        # print(loss.shape)
        return loss[0]


class CLIPVisualLoss(torch.nn.Module):
    """CLIP visual loss, a-la CLIPAsso"""

    def __init__(
        self,
        input_size=224,
        rgb=True,
        clipag=False,
        clip_model="ViT-B-32",
        semantic_w=0.1,
        geometric_w=1.0,
        crop_scale=0.9,
        distortion_scale=0.5,
        vis_metric="mse",
        layer_weights=[(2, 1.0), (3, 1.0)],
    ):
        import open_clip
        from . import fs

        super().__init__()

        self.semantic_w = semantic_w
        self.geometric_w = geometric_w

        if clipag:
            clip_model = "CLIPAG"
        model, preprocess, tokenizer, self.input_size = load_clip_model(clip_model)
        self.crop_scale = crop_scale
        self.distortion_scale = distortion_scale
        self.layer_weights = layer_weights
        self.rgb = rgb
        self.vis_metric = vis_metric
        self.model = model  # .to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.normalize = transforms.Compose(
            [
                preprocess.transforms[0],  # Resize
                # preprocess.transforms[1],  # CenterCrop
                preprocess.transforms[-1],  # Normalize
            ]
        )

        # try:
        #     self.input_size = preprocess.transforms[0].size[0]
        # except TypeError:
        #     self.input_size = preprocess.transforms[0].size

        self.feature_maps = OrderedDict()

        try:
            for i in range(12):
                model.visual.transformer.resblocks[i].register_forward_hook(
                    self.make_hook(i)
                )
        except AttributeError as e:
            print("Resblocks not present attempting trunk")
            try:
                for i in range(12):
                    model.visual.trunk.blocks[i].register_forward_hook(
                        self.make_hook(i)
                    )
            except AttributeError as e:

                flat_idx = 0
                for stage in model.visual.trunk.stages:
                    for block in stage.blocks:
                        block.register_forward_hook(self.make_hook(flat_idx))
                        flat_idx += 1
                print("ConvNetXt registered ", flat_idx, "hooks")

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.feature_maps[name] = output.permute(1, 0, 2)
            else:
                self.feature_maps[name] = output

        return hook

    def encode_image(self, image):
        self.feature_maps = OrderedDict()

        fc = self.model.encode_image(
            image
        )  # self.transform(image)) #.permute(0, 3, 1, 2)))
        feature_maps = self.feature_maps
        return fc, feature_maps

    @property
    def clip_norm_(self):
        return transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, x, y, num_aug=4):
        x = to_batch(x, self.rgb)
        y = to_batch(y, self.rgb).to(dtype=x.dtype)
        im_res = self.input_size  # x.shape[-1]
        if geom.is_number(self.crop_scale):
            crop_scale = (self.crop_scale, self.crop_scale)
        else:
            crop_scale = self.crop_scale
        # init augmentations
        augment_list = []
        if self.distortion_scale > 0:
            augment_list.append(
                transforms.RandomPerspective(
                    fill=1, p=1.0, distortion_scale=self.distortion_scale
                )  # 0.5)
            )
        augment_list.append(
            transforms.RandomResizedCrop(im_res, scale=crop_scale, ratio=(1.0, 1.0))
        )
        augment_list.append(
            transforms.GaussianBlur(
                kernel_size=21, sigma=(0.01, 2.0)
            )  # Example: 5x5 kernel
        )
        augment_list.append(self.clip_norm_)  # CLIP Normalize
        # compose augmentations
        augment_compose = transforms.Compose(augment_list)

        # make augmentation pairs
        x_augs, y_augs = [self.normalize(x)], [self.normalize(y)]
        # repeat N times
        for n in range(num_aug):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(x_augs, dim=0)
        ys = torch.cat(y_augs, dim=0)
        self.x_augs = [xa.detach().cpu().numpy()[0, 0, :, :] for xa in x_augs]
        self.y_augs = [ya.detach().cpu().numpy()[0, 0, :, :] for ya in y_augs]

        fc_true, fm_true = self.encode_image(ys)
        fc_pred, fm_pred = self.encode_image(xs)

        fc_loss = (1 - torch.cosine_similarity(fc_true, fc_pred, dim=1)).mean()

        fm_loss = 0
        for (
            i,
            w,
        ) in self.layer_weights:  # [3, 5]: #2,3]: #3, 5]: #1,2,3]: #[3, 4]: #2, 3]:
            if self.vis_metric == "mse" or self.vis_metric == "L2":
                fm_loss += w * torch.square(fm_true[i] - fm_pred[i]).mean()
            elif self.vis_metric == "L1":
                fm_loss += w * torch.abs(fm_true[i] - fm_pred[i]).mean()
            else:
                fm_loss += (
                    w
                    * (
                        1 - torch.cosine_similarity(fm_true[i], fm_pred[i], dim=1)
                    ).mean()
                )

        total_loss = self.semantic_w * fc_loss + self.geometric_w * fm_loss
        # print('Clip total', total_loss.item())
        return total_loss


def image_augmentation_clip(size, args=["Af", "Ji"]):
    """See https://pytorch.org/vision/stable/transforms.html"""
    items = []
    for item in args:
        if item == "Pe":
            items.append(
                transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.2)
            )
        elif item == "Re":
            items.append(transforms.RandomResizedCrop(size, scale=(0.8, 0.95)))
        elif item == "Ji":
            items.append(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                )
            )
        elif item == "Af":
            items.append(
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), shear=2, fill=1
                )
            )

    augment_trans = transforms.Compose(items)
    return augment_trans

    augment_trans = transforms.Compose(
        [
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.2),
            transforms.RandomResizedCrop(224, scale=(0.8, 0.95)),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),  # , p=0.7),
        ]
    )
    return augment_trans


def load_clip_model(model_name):
    import open_clip

    if model_name in cfg.clip_models:
        print(model_name, "already loaded")
        return cfg.clip_models[model_name]

    if model_name == "CLIPAG":
        from . import fs

        print("Downlading CLIPAG")
        url = "https://zenodo.org/records/10446026/files/CLIPAG_ViTB32.pt?download=1"
        path = "./CLIPAG.pt"
        fs.download_file_once(url, path)
        pretrained = path
        model_name = "ViT-B-32"
    elif model_name == "EGCLIP":
        model_name = "convnext_xxlarge"
        pretrained = "/home/danielberio/Downloads/model_epoch_12.pt"
    else:
        pretrained_map = {
            "ViT-H/14-quickgelu": "dfn5b",
            "ViT-B-32": "laion2b_s34b_b79k",
            "ViT-B-16-SigLIP-384": "webli",
            "ViT-L-16-SigLIP-256": "webli",
            "ViT-L-16-SigLIP-384": "webli",
            "ViT-SO400M-14-SigLIP-384": "webli",
            "ViT-SO400M-14-SigLIP": "webli",  # Ok
            "ViT-SO400M/14": "webli",
            "ViT-L-14": "laion2b_s32b_b82k",  # Good for sketches?
            "ViT-L-14-quickgelu": "metaclip_fullcc",
            "ViT-g-14": "laion2b_s34b_b88k",
            "ViT-B-16": "datacomp_xl_s13b_b90k",
            "EVA02-L-14": "merged2b_s4b_b131k",
            "ViT-H-14-CLIPA": "datacomp1b",
            "ViT-H-14-378-quickgelu": "dfn5b",  # No mem
            "ViT-L-14-CLIPA-336": "datacomp1b",
            "ViT-H-14-quickgelu": "metaclip_fullcc",  # Good, but slow
            "ViT-H-14-378-quickgelu": "dfn5b",
            "ViT-B-16-SigLIP-384": "webli",
            "ViT-H-14-quickgelu": "metaclip_fullcc",
            "ViT-L-14-CLIPA": "datacomp1b",
            "EVA02-L-14": "merged2b_s4b_b131k",
            "ViT-B-32-256": "datacomp_s34b_b86k",
        }
        pretrained = pretrained_map[model_name]
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="amp",
        weights_only=False,  # Breaks otherwise
        device=device,
    )
    try:
        input_size = preprocess.transforms[0].size[0]
    except TypeError:
        input_size = preprocess.transforms[0].size
    print("Input size is ", input_size)
    tokenizer = open_clip.get_tokenizer(model_name)
    cfg.clip_models[model_name] = (model, preprocess, tokenizer, input_size)
    return model, preprocess, tokenizer, input_size


# Based on SciPy
# https://github.com/heroxbd/waveform-analysis/blob/ec6f50f09923cdab74e85e050245c7e1622faa9d/loss.py#L16
def wasserstein_loss(a, b):
    return cdf_loss(a, b, 1)


def cdf_loss(tensor_a, tensor_b, p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)), dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(
            torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1)
        )
    else:
        cdf_distance = torch.pow(
            torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p), dim=-1),
            1 / p,
        )

    cdf_loss = cdf_distance.mean()
    return cdf_loss
