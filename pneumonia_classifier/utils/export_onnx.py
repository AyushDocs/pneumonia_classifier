import torch
import torch.onnx

from pneumonia_classifier.ml.model.arch import PneumoniaEnsemble


def export_to_onnx(model_path="notebooks/mlruns/2/models/m-75861528c062408c86d30fcafe7c1bb0/artifacts/data/model.pth", output_path="notebooks/mlruns/2/models/m-75861528c062408c86d30fcafe7c1bb0/artifacts/data/ensemble.onnx"):
    try:
        # If the file is a state_dict
        model = PneumoniaEnsemble()
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    except Exception:
        # If the file is an entire model object
        model = torch.load(model_path, map_location="cpu", weights_only=False)

    model.eval()

    # Define dummy input matching the 224x224 RGB image tensors
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Done!")

if __name__ == "__main__":
    export_to_onnx()
if __name__ == "__main__":
    export_to_onnx()
