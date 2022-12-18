from resnet20 import resnet20_add
from resnet20_cnn import resnet20
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


model = resnet20()
inputs = torch.randn(256, 3, 32, 32)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=50))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=50))

# from thop import profile, clever_format
# macs, params = profile(model, inputs=(inputs, ))
# macs, params = clever_format([macs, params], "%.3f")
# print('macs :', macs)
# print('params :', params)