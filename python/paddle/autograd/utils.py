# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle


def _tensors(ts, name):
    if isinstance(ts, (list, tuple)):
        assert len(ts) > 0, f"{name} connot be empty"
        for each_t in ts:
            assert (
                isinstance(each_t, paddle.Tensor) or each_t is None
            ), f"Elements of {name} must be paddle.Tensor or None"
        return list(ts)
    else:
        assert isinstance(ts, paddle.Tensor), f"{name} must be Tensor"
        return [ts]


def _stack_tensor_or_return_none(origin_list):
    assert len(origin_list) > 0, "Can't not stack an empty list"
    return paddle.stack(
        origin_list, axis=0) if isinstance(origin_list[0],
                                           paddle.Tensor) else None


def _replace_none_with_zero_tensor(t, spec_t):
    if t is not None:
        return t
    zero_t = paddle.zeros(shape=spec_t.shape, dtype=spec_t.dtype)
    zero_t.stop_gradient = spec_t.stop_gradient
    return zero_t
