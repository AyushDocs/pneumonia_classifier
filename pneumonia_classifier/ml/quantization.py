import torch
import torch.quantization


def quantize_model_int8(model, calibration_loader=None):
    """
    Performs Static Quantization (float32 -> int8) on the given PyTorch model.
    """
    model.eval()

    # 1. Attach quantization/dequantization stubs
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)

    # 2. Calibrate (if calibration_loader provided)
    if calibration_loader:
        with torch.no_grad():
            for images, _ in calibration_loader:
                model_prepared(images)

    # 3. Convert to quantized version
    model_quantized = torch.quantization.convert(model_prepared)

    return model_quantized

def save_quantized_model(model, path):
    """
    Saves the quantized model state.
    """
    torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), path)
