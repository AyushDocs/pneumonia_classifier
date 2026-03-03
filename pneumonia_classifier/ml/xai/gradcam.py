import cv2
import numpy as np
import torch


def generate_gradcam(model, input_tensor, target_layer):
    activations = []
    gradients = []

    def save_activations(module, input, output):
        activations.append(output)

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(save_activations)
    h2 = target_layer.register_full_backward_hook(save_gradients)

    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()

    model.zero_grad()
    one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float).to(input_tensor.device)
    one_hot[0][pred_idx] = 1
    output.backward(gradient=one_hot)

    h1.remove()
    h2.remove()

    grads = gradients[0].cpu().data.numpy()
    target = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(2, 3))[0]

    cam = np.zeros(target.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * target[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-9)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    return heatmap


