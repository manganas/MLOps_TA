from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import torch

from mnist.models.model import SimpleCNN

import hydra

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):

    img_size = cfg.experiment.img_size
    num_classes = cfg.experiment.num_classes
    batch_size = cfg.experiment.batch_size

    model = SimpleCNN(28, 10)
    inputs = torch.randn(batch_size, 1, img_size, img_size)


    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
        model(inputs)


    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print('*'*50)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

    prof.export_chrome_trace("trace.json")

if __name__=='__main__':
    main()
