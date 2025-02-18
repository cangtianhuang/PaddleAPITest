from .api_config import TensorConfig, APIConfig, analyse_configs

import re
import collections
import paddle
import numpy
import math
import json
import torch
import paddle
import inspect
from .base import APITestBase
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]

api_config_accuracy_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_accuracy_error.txt", "a")
api_config_paddle_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_paddle_error.txt", "a")
api_config_pass = open(DIR_PATH+"/tester/api_config/test_log/api_config_pass.txt", "a")
api_config_torch_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_torch_error.txt", "a")

class APITestAccuracy(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    def test(self):
        if self.need_skip():
            return

        if not self.ana_api_info():
            return

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            with torch.no_grad():
                if not self.gen_torch_input():
                    return
                torch_output = self.torch_api(*tuple(self.torch_args), **self.torch_kwargs)
                self.clear_torch_tensor()
        except Exception as err:
            print("[torch error]", self.api_config.config, "\n", str(err))
            torch_output = None
            api_config_torch_error.write(self.api_config.config+"\n")
            api_config_torch_error.flush()
            return

        try:
            with paddle.no_grad():
                if not self.gen_paddle_input():
                    return
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                self.clear_paddle_tensor()
        except Exception as err:
            print("[paddle error]", self.api_config.config, "\n", str(err))
            torch_output = None
            paddle_output = None
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err))
            torch_output = None
            paddle_output = None
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            return

        if isinstance(paddle_output, paddle.Tensor):
            if isinstance(torch_output, torch.Tensor):
                try:
                    if paddle_output.dtype == paddle.bfloat16:
                        paddle_output = paddle.cast(paddle_output, dtype="float32")
                        torch_output = torch_output.to(dtype=torch.float32)
                    self.np_assert_accuracy(paddle_output.numpy(), torch_output.cpu().numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err))
                    torch_output = None
                    paddle_output = None
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    api_config_accuracy_error.flush()
                    return
            else:
                print("[output type diff error]", self.api_config.config)
        elif isinstance(paddle_output, (list, tuple)):
            if isinstance(paddle_output, tuple):
                paddle_output = list(paddle_output)
            if not isinstance(torch_output, (list, tuple)):
                print("[output type diff error]", self.api_config.config)
                torch_output = None
                paddle_output = None
                return
            if isinstance(torch_output, tuple):
                torch_output = list(torch_output)
            if len(paddle_output) != len(torch_output):
                print("[output type diff error]", self.api_config.config)
                torch_output = None
                paddle_output = None
                return
            for i in range(len(paddle_output)):
                try:
                    if paddle_output.dtype == paddle.bfloat16:
                        paddle_output = paddle.cast(paddle_output, dtype="float32")
                        torch_output = torch_output.to(dtype=torch.float32)
                    self.np_assert_accuracy(paddle_output[i].numpy(), torch_output[i].cpu().numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err))
                    torch_output = None
                    paddle_output = None
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    api_config_accuracy_error.flush()
                    return
        torch_output = None
        paddle_output = None
        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
        api_config_pass.flush()
  
