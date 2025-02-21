// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/kernel/pten_kernels.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/kernels/math_kernel.h"

using infrt::host_context::Attribute;

namespace infrt {
namespace kernel {

void RegisterPtenKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("pd_cpu.add.float32",
                      INFRT_KERNEL(pten::AddKernel<float, pten::CPUContext>));
  registry->AddKernel("pd_cpu.add.int32",
                      INFRT_KERNEL(pten::AddKernel<int, pten::CPUContext>));
}

}  // namespace kernel
}  // namespace infrt
