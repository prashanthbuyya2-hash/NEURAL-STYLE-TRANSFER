import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Preprocessing ---
def load_image(img, max_size=400, shape=None):
    image = img.convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return Image.fromarray((image * 255).astype(np.uint8))

# --- Feature Extraction ---
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# --- Style Transfer Function ---
def style_transfer(content_img, style_img, steps=500, content_weight=1e4, style_weight=1e2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content = load_image(content_img).to(device)
    style = load_image(style_img, shape=content.shape[-2:]).to(device)

    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1.0,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}

    optimizer = optim.Adam([target], lr=0.003)

    for step in range(steps):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return im_convert(target)

# --- Gradio Interface ---
demo = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image"),
        gr.Image(type="pil", label="Style Image"),
        gr.Slider(100, 2000, value=500, label="Training Steps"),
        gr.Slider(1e3, 1e5, value=1e4, label="Content Weight"),
        gr.Slider(1e1, 1e3, value=1e2, label="Style Weight")
    ],
    outputs=gr.Image(type="pil", label="Stylized Output"),
    title="🎨 Neural Style Transfer",
    description="Upload a content image and a style image to blend them together using PyTorch and VGG19."
)

demo.launch()

