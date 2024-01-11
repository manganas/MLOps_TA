import torch
import torch.utils.benchmark as benchmark
import timeit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained=True)

model.eval()

model_jit = torch.jit.script(model)
model_jit.save("models/deployable_model.pt")

model.to(device)
model_jit.to(device)

img_in = torch.randn(64, 3, 224, 224)

unscripted_top5_indices = model(img_in).topk(5).indices
scripted_top5_indices = model_jit(img_in).topk(5).indices

assert torch.allclose(
    unscripted_top5_indices, scripted_top5_indices
), "The output of the models is not the same"


## Benchmarking
