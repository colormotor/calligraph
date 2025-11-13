#!/usr/bin/env python3

import torch
from transformers import AutoProcessor, AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoModelForUniversalSegmentation
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from . import config

device = config.device

def find_segments(im, kind='instance'):
    if type(im) == np.ndarray:
        im = Image.fromarray(im)

    if kind == 'panoptic':
        processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        fn = processor.post_process_panoptic_segmentation
    elif kind == 'semantic':

        processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco")
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-semantic")
        fn = processor.post_process_semantic_segmentation
    else: # 'instance'
        # "facebook/mask2former-swin-large-coco-instance")
        name = "facebook/mask2former-swin-large-coco-instance" #"facebook/mask2former-swin-large-coco" #-instance"
        #name = "adirik/maskformer-swin-base-sceneparse-instance"
        processor = AutoImageProcessor.from_pretrained(name, ignore_mismatched_sizes=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(name, ignore_mismatched_sizes=True)
        #processor = AutoImageProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        #model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)

        fn = processor.post_process_instance_segmentation
    inputs = processor(images=im, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = fn(outputs, target_sizes=[im.size[::-1]])[0]

    if kind=='semantic':
        return results

    segments = []
    for segment in results['segments_info']     :
        mask = (results['segmentation'].numpy() == segment['id'])
        label = model.config.id2label[segment['label_id']]
        segment['label'] = label
        print(label)
        visual_mask = (mask * 255).astype(np.uint8)
        visual_mask = Image.fromarray(visual_mask)
        segment['visual_mask'] = visual_mask
        segment['mask'] = mask
        segments.append(segment)

    return results['segmentation'], len(segments)

class MySequential(torch.nn.Sequential):
    def __init__(self, *args):
        super(MySequential, self).__init__(*args)
        self.register_buffer('gradients', None)

    def forward(self, x):
        # Clear the gradients buffer
        self.gradients = None

        x = self.grad_heatmap(x)
        return x

    def save_gradient(self, grad):
        self.gradients = grad

    def grad_heatmap(self, x):

        mask_probs, class_queries_logits = x

        num_classes = class_queries_logits.shape[-1] - 1

        uniform_class_queries_logits = torch.full_like(class_queries_logits, (num_classes - 1) / num_classes)

        uniform_class_weight, _ = torch.nn.functional.softmax(uniform_class_queries_logits, dim=-1).max(-1)

        class_weight, pred = torch.nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        elems, counts = torch.unique(pred, return_counts=True)

        # normalization
        for batch in range(class_weight.shape[0]):
            for i, c in enumerate(elems):
                max_mask = torch.where(pred[batch] == c)
                class_weight[batch][max_mask] /= counts[i]

        weight = mask_probs * class_weight.view(mask_probs.shape[0], mask_probs.shape[1], 1, 1)

        del uniform_class_weight, uniform_class_queries_logits, num_classes, class_weight

        weight.register_hook(self.save_gradient)

        return weight

def segmentation(image):
    processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
    print("Loaded model, processing")
    model.to(device)

    features = {}

    panoptic_inputs = processor(images=[image], task_inputs=["panoptic"], return_tensors="pt")
    panoptic_inputs.to(device)

    print("Results")
    with torch.no_grad():
        outputs = model(**panoptic_inputs)

    del model
    del processor
    del panoptic_inputs

    print(f"Finished model inference. Starting postprocessing")

    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    batch_size = mask_probs.shape[0]
    num_queries = mask_probs.shape[1]
    num_classes = class_queries_logits.shape[-1] - 1
    height = mask_probs.shape[2]
    width = mask_probs.shape[3]

    model_input_mask_probs = mask_probs.clone().detach().requires_grad_(True)

    model_mini = MySequential()

    print(f"Compute gradient norms")

    # WITH GRAD ENABLED
    pred_scores_simple_head = model_mini((model_input_mask_probs, class_queries_logits))
    pred_scores, pred_labels = torch.nn.functional.softmax(class_queries_logits, dim=-1).max(-1)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(pred_scores_simple_head, mask_probs)
    model_mini.zero_grad()
    loss.backward()

    gradients = model_mini.gradients
    norm = torch.sqrt(torch.norm(gradients, dim=1, p=2)).cpu()

    del model_mini
    del model_input_mask_probs

    scaled_OoD_feature_detector = torch.nn.functional.interpolate(
        norm.unsqueeze(0), size=image.size[::-1], mode="bilinear", align_corners=False
    )[0]

    # this should now measure the average norm of the gradients according to the
    # model’s predicted distribution reweighted to reflect the model output bias
    # Same meaure as https://arxiv.org/pdf/2404.10124 (REGrad) which takes the
    # l2 norm and takes the squareroot of https://arxiv.org/pdf/2205.10439
    # (EXGRAG) to mitigate overconfident results, where a high model probability
    # overshadows the contributions from gradients of other classes, potentially
    # leading to an overemphasis. This might neglect important insights about
    # the model’s decision boundaries and uncertain regions indicated by other
    # class gradient We apply a rescaling of the logits before taking the
    # gradinent based on the predicted labels's prominence across the features
    # representing it to approximate a mitigation for model training bias,
    # without calculating mean and var across the whole training set
    # (https://arxiv.org/pdf/2107.11264)

    OoD_feature_detector_inter = torch.mean(scaled_OoD_feature_detector, dim=0)
    OoD_feature_detector_inter_min, OoD_feature_detector_inter_max = OoD_feature_detector_inter.min(), OoD_feature_detector_inter.max()
    #print('OoD Minmax:', OoD_feature_detector_inter_min, OoD_feature_detector_inter_max)
    OoD_feature_detector_inter_normalized = (OoD_feature_detector_inter - OoD_feature_detector_inter_min)/(OoD_feature_detector_inter_max - OoD_feature_detector_inter_min)
    return OoD_feature_detector_inter_normalized

def saliency(image):

    # https://huggingface.co/mattmdjaga/segformer_b2_clothes/tree/main
    #processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    #model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # https://huggingface.co/jonathandinu/face-parsing
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    inputs = processor(images=np.array(image), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    del model
    del inputs

    face_prediction = outputs.logits.cpu() # (batch_size, config.num_labels, logits_height, logits_width)

    # interpolate to original size
    face_prediction = torch.nn.functional.interpolate(
        face_prediction,
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    face_prediction_segmented = face_prediction.argmax(dim=1)[0]
    face_prediction = torch.median(face_prediction[0], dim=0).values # mean without .values!
    face_prediction_min, face_prediction_max = face_prediction.min(), face_prediction.max()
    face_prediction_normalized = (face_prediction - face_prediction_min)/(face_prediction_max - face_prediction_min)
    return face_prediction_normalized

def find_segments(im, kind='instance'):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoModelForUniversalSegmentation
    
    if type(im) == np.ndarray:
        im = Image.fromarray(im)

    if kind == 'panoptic':
        processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        fn = processor.post_process_panoptic_segmentation
    elif kind == 'semantic':

        processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco")
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-semantic")
        fn = processor.post_process_semantic_segmentation
    else: # 'instance'
        # "facebook/mask2former-swin-large-coco-instance")
        name = "facebook/mask2former-swin-large-coco-instance" #"facebook/mask2former-swin-large-coco" #-instance"
        #name = "adirik/maskformer-swin-base-sceneparse-instance"
        processor = AutoImageProcessor.from_pretrained(name, ignore_mismatched_sizes=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(name, ignore_mismatched_sizes=True)
        #processor = AutoImageProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        #model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)

        fn = processor.post_process_instance_segmentation
    inputs = processor(images=im, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = fn(outputs, target_sizes=[im.size[::-1]])[0]

    if kind=='semantic':
        return results

    segments = []
    for segment in results['segments_info']:

        mask = (results['segmentation'].numpy() == segment['id'])
        label = model.config.id2label[segment['label_id']]
        segment['label'] = label
        # print(label)
        visual_mask = (mask * 255).astype(np.uint8)
        visual_mask = Image.fromarray(visual_mask)
        segment['visual_mask'] = visual_mask
        segment['mask'] = mask
        segments.append(segment)

    return segments


def xdog(im, k=10, gamma=0.98, phi=200, eps=-0.1, sigma=0.8, binarize=True):
    # From https://github.com/yael-vinker/CLIPasso/blob/main/models/painter_params.py
    from scipy.ndimage.filters import gaussian_filter
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu

    if type(im) != np.ndarray:
        im = np.array(im)
    if len(im.shape)>2 and im.shape[2] == 3:
        im = rgb2gray(im)
    imf1 = gaussian_filter(im, sigma)
    imf2 = gaussian_filter(im, sigma * k)
    imdiff = imf1 - gamma * imf2
    imdiff = (imdiff < eps) * 1.0  + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
    imdiff -= imdiff.min()
    imdiff /= imdiff.max()
    if binarize:
        th = threshold_otsu(imdiff)
        imdiff = imdiff >= th
    imdiff = imdiff.astype('float32')
    return imdiff


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: torch.nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad

def clip_saliency(image, prompt='', model='ViT-B/32', layers=[]):
    # https://arxiv.org/abs/2304.05653
    from skimage.transform import resize
    from .contrib.CLIPExplain import clip
    
    model, preprocess = clip.load(model, device=device, jit=False)
    text = clip.tokenize([prompt]).to(device)
    img = preprocess(image).unsqueeze(0).to(device)
    fn = interpret
    if prompt:
        fn = interpret2
    res = resize(fn(img, text, model, layers), (image.height, image.width))
    del model
    del preprocess
    return res

def interpret(image, texts, model, layers=[], start_layer=-1):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images) #, mode="saliency")
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    #print(image_attn_blocks)
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1
    print('n blocks:', len(image_attn_blocks))
    print('start layer', start_layer)
    if not layers:
        layers = range(len(image_attn_blocks))
    for i in layers: #, blk in enumerate(image_attn_blocks):
        #if i < start_layer:
        #    continue

        blk = image_attn_blocks[i]
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        #print('shp', cam.shape)
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)
    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance

def interpret2(image, texts, model, layers=[]): #start_layer=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    #image_features = model.encode_image(images, mode='saliency')
    logits_per_image, logits_per_text = model(images, texts) #, mode='saliency')
    #logits_per_image = image_features/image_features.norm(dim=-1, keepdim=True)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    # if start_layer == -1:
    #   # calculate index of last layer
    #   start_layer = len(image_attn_blocks) - 1

    if not layers:
        layers = range(len(image_attn_blocks))

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    cams = []
    for i in layers: #, blk in enumerate(image_attn_blocks):
        #if i < start_layer:
        #  continue
        blk = image_attn_blocks[i]
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        cams.append(cam)
        R = R + torch.bmm(cam, R)
    # print('R', R.shape)
    image_relevance = R[:, 0, 1:]

    # cams_avg = torch.cat(cams) # 12, 50, 50
    # print('cams_avg', cams_avg.shape)
    # cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    # image_relevance = cams_avg.mean(dim=0).unsqueeze(0)

    dim = int(image_relevance.numel() ** 0.5)
    # print('dim is', dim)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance

