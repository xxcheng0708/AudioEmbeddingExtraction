# coding:utf-8
"""
    Created by cheng star at 2022/1/22 22:00
    @email : xxcheng0708@163.com
"""
import torch
import torchvision
from torch import nn
import onnxruntime
import numpy as np
import time
from models.model_resnet import AudioEmbeddingModel


def pytorch_2_onnx(model_path, export_model_path):
    batch_size = 8
    model = AudioEmbeddingModel(input_dimension=64, out_dimension=128, model_name="resnet18")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    input_names = ["input_data"]
    output_names = ["output"]
    dummy_input = torch.rand(batch_size, 1, 64, 1001)

    torch.onnx.export(model, dummy_input, export_model_path, verbose=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes={
                          "input_data": {0: "batch_size"},
                          "output": {0: "batch_size"}
                      })
    return export_model_path


def onnx_test(export_model_path):
    session = onnxruntime.InferenceSession(export_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Input Name:", input_name)
    print("Output Name:", output_name)

    input_data = np.random.randn(64, 1001)
    input_data = input_data[np.newaxis, np.newaxis, :, :]
    print("input_data: ", input_data.shape)

    total_time = 0
    for _ in range(100):
        start_time = time.time()
        result = session.run(None, {input_name: input_data})
        result = result[0]
        print(result.shape)

        end_time = time.time()
        duration_time = int((end_time - start_time) * 1000)
        total_time += duration_time
    avg_time = total_time / 100
    print("avg time: {} ms".format(avg_time))
