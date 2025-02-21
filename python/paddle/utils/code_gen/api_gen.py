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

import os
import yaml
import argparse

import gen_utils


class API:
    prefix_tensor_name = 'dense_'

    def __init__(self, api_item_yaml):
        self.api = api_item_yaml['api']
        # args:
        #   inputs:
        #     names : [], list of input names
        #   attrs:
        #     names : [], list of attribute names
        #     attr_info : { attr_name : (type, default_values)}
        self.args = gen_utils.parse_args(self.api, api_item_yaml['args'])
        self.out_type_list, _ = gen_utils.parse_output(self.api,
                                                       api_item_yaml['output'])
        self.return_type = self.out_type_list[0] if len(
            self.out_type_list) == 1 else "std::tuple<" + ",".join(
                self.out_type_list) + ">"

        self.is_base_api = True
        if 'invoke' in api_item_yaml:
            self.is_base_api = False
            self.invoke = api_item_yaml['invoke']
        else:
            self.kernel = api_item_yaml['kernel']
            if 'backend' not in self.kernel or len(self.kernel['backend']) == 0:
                self.kernel['backend'] = None
            if 'layout' not in self.kernel or len(self.kernel['layout']) == 0:
                self.kernel['layout'] = None
            if 'data_type' not in self.kernel or len(self.kernel[
                    'data_type']) == 0:
                self.kernel['data_type'] = None
            if 'param' not in self.kernel:
                self.kernel['param'] = None

            self.infer_meta = api_item_yaml['infer_meta']
            if 'param' not in self.infer_meta:
                self.infer_meta['param'] = None

    def gene_api_declaration(self):
        return f"""
PADDLE_API {self.return_type} {self.api}({self.args['args_declare']});
"""

    def gene_output(self, output_type_list):
        kernel_output = ""
        output_create = ""

        if len(output_type_list) == 1:
            kernel_output = 'dense_out'
            output_create = f"""
  {self.return_type} out;
  auto dense_out = SetKernelOutput(out_meta, kernel_backend, &out);"""

        elif len(output_type_list) > 1:
            output_create = f"""
  {self.return_type} out;"""

            for i in range(len(output_type_list)):
                kernel_output = kernel_output + f'dense_out_{i}, '
                output_create = output_create + f"""
  auto dense_out_{i} = SetKernelOutput(std::get<{i}>(out_meta), kernel_backend, &std::get<{i}>(out));"""

            kernel_output = kernel_output[:-2]
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api))

        return kernel_output, output_create

    def gene_api_code(self):
        if self.is_base_api:
            input_tensors, kernel_args = gen_utils.get_kernel_args(
                self.args['inputs']['names'], self.args['attrs'],
                self.kernel['param'])
            outputs_args, output_create = self.gene_output(self.out_type_list)
            return f"""
PADDLE_API {self.return_type} {self.api}({self.args["args_define"]}) {{
{gen_utils.gene_kernel_select(self.api, self.args['inputs']['names'], self.args['attrs'], self.kernel)}

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
{input_tensors}
{gen_utils.gene_infer_meta(self.args['inputs']['names'], self.args['attrs']['names'], self.infer_meta)}
{output_create}

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::{self.api}_kernel>();
  (*kernel_fn)({kernel_args}, {outputs_args});

  return out;
}}
"""

        else:
            return f"""
PADDLE_API {self.return_type} {self.api}({self.args["args_define"]}) {{
  return {self.invoke};
}}
"""


def header_include():
    return """
#include <tuple>

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/kernel_signature.h"
#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/api_utils.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/infermeta/binary.h"
#include "paddle/pten/infermeta/multiary.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/declarations.h"
"""


def api_register():
    return """
PT_REGISTER_API(Math);
"""


def api_namespace():
    return ("""
namespace paddle {
namespace experimental {

""", """

}  // namespace experimental
}  // namespace paddle
""")


def generate_api(api_yaml_path, header_file_path, source_file_path):

    with open(api_yaml_path, 'r') as f:
        apis = yaml.load(f, Loader=yaml.FullLoader)
    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/pten/api/include/api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])

    for api in apis:
        api_code = API(api)
        print(api_code.gene_api_declaration())
        header_file.write(api_code.gene_api_declaration())
        source_file.write(api_code.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])
    source_file.write(api_register())

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files')
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        default='python/paddle/utils/code_gen/api.yaml')
    parser.add_argument(
        '--api_header_path',
        help='output of generated api header code file',
        default='paddle/pten/api/include/api.h')

    parser.add_argument(
        '--api_source_path',
        help='output of generated api source code file',
        default='paddle/pten/api/lib/api.cc')

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    header_file_path = options.api_header_path
    source_file_path = options.api_source_path

    generate_api(api_yaml_path, header_file_path, source_file_path)


if __name__ == '__main__':
    main()
