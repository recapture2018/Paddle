// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"

DECLARE_bool(use_mkldnn);

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

const framework::Tensor* GetTensorFromVar(const framework::Variable& var);

template <typename VarType>
static void SetForwardDataTypeOfGradVar(const std::shared_ptr<VarType>& var);

template <>
void SetForwardDataTypeOfGradVar<VariableWrapper>(
    const std::shared_ptr<VariableWrapper>& var) {
  if (var->HasGradVar()) {
    auto grad_var = var->GetGradVar();
    VLOG(6) << "Set grad var (" << grad_var->Name() << ")'s forward dtype to ("
            << framework::DataTypeToString(var->DataType()) << ").";
    grad_var->SetForwardDataType(var->DataType());
  }
}

template <>
void SetForwardDataTypeOfGradVar<VarBase>(const std::shared_ptr<VarBase>& var) {
  if (var->HasGradVar()) {
    auto& shared_var = var->SharedVar();
    SetForwardDataTypeOfGradVar<VariableWrapper>(shared_var);
  }
}

extern const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<paddle::imperative::VarBase>& var);
extern const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<VariableWrapper>& var);

template <typename VarType>
std::shared_ptr<NameVarMap<VarType>> PrepareData(
    const framework::OperatorWithKernel& op, const NameVarMap<VarType>& ins,
    const framework::OpKernelType& expected_kernel_key) {
  std::shared_ptr<NameVarMap<VarType>> tmp_ins_ptr = nullptr;
  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) {
      auto& var_base = name_pair.second[i];
      SetForwardDataTypeOfGradVar(var_base);
      const auto* tensor = GetTensorFromVar(var_base->Var());
      if (tensor && tensor->IsInitialized()) {
        auto kernel_type_for_var = op.GetKernelTypeForVar(
            name_pair.first, *tensor, expected_kernel_key);
        if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
          continue;
        } else {
          VLOG(3) << "Transform Variable " << var_base->Name() << " from "
                  << kernel_type_for_var << " to " << expected_kernel_key;

          if (GetVariableWrapper(var_base)->hasCacheKey(expected_kernel_key)) {
            VLOG(3) << "Hit variable_wrapper cache: key="
                    << expected_kernel_key;
            std::shared_ptr<VariableWrapper> cache_var =
                GetVariableWrapper(var_base)->getCacheValue(
                    expected_kernel_key);
            if (tmp_ins_ptr == nullptr) {
              tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
            }

            const auto* tensor = GetTensorFromVar(cache_var->Var());
            auto tmp_var = std::make_shared<VarType>(var_base->Name());
            tmp_var->SetType(var_base->Type());
            SetTensorToVariable(cache_var->Var(), *tensor,
                                tmp_var->MutableVar());
            (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
          } else {
            framework::Tensor out;
            TransformData(expected_kernel_key, kernel_type_for_var, *tensor,
                          &out);
            if (NeedTransformDataType(kernel_type_for_var,
                                      expected_kernel_key)) {
              // To avoid NameVarMap copy construction overhead in general
              // scenarios, if inplace transformed, return original input
              // directly
              if (tmp_ins_ptr == nullptr) {
                tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
              }
              auto tmp_var = std::make_shared<VarType>(var_base->Name());
              tmp_var->SetType(var_base->Type());
              SetTensorToVariable(var_base->Var(), out, tmp_var->MutableVar());
              (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;

              GetVariableWrapper(var_base)->setCacheValue(
                  expected_kernel_key, GetVariableWrapper(tmp_var));
              VLOG(3) << "Set cache to variable_wrapper: key="
                      << expected_kernel_key;
            } else {
              // if dtype is same, transform inplace will not change the
              // original
              // value, transform inplace to avoid multiple copy
              SetTensorToVariable(var_base->Var(), out, var_base->MutableVar());
            }
          }
        }
      }
    }
  }
  return tmp_ins_ptr;
}

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const framework::OpKernelType& kernel_type,
             const framework::OperatorWithKernel::OpKernelFunc& func,
             platform::DeviceContext* dev_ctx);

  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const framework::OpKernelType& kernel_type,
             const framework::KernelSignature& kernel_signature,
             const pten::Kernel& pt_kernel, platform::DeviceContext* dev_ctx);

  static PreparedOp Prepare(const NameVarMap<VarBase>& ins,
                            const NameVarMap<VarBase>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  static PreparedOp Prepare(const NameVarMap<VariableWrapper>& ins,
                            const NameVarMap<VariableWrapper>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<VarBase>& in, const NameVarMap<VarBase>& out,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<VariableWrapper>& ins,
           const NameVarMap<VariableWrapper>& outs,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  const framework::OpKernelType& kernel_type() const { return kernel_type_; }

 private:
  const framework::OperatorBase& op_;
  const framework::RuntimeContext& ctx_;
  framework::OpKernelType kernel_type_;
  framework::OperatorWithKernel::OpKernelFunc func_;
  platform::DeviceContext* dev_ctx_;
  // NOTE(chenweihang): Similar op members are used to adapt to
  // new pten kernel, if there is a better design in the future,
  // we may polish the implementation here
  bool run_pten_kernel_{false};
  framework::KernelSignature pt_kernel_signature_;
  pten::Kernel pt_kernel_;
};

const inline framework::Attribute& GetAttr(
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs, const std::string& name) {
  auto it = attrs.find(name);
  bool found = it != attrs.end();
  if (!found) {
    it = default_attrs.find(name);
    found = it != default_attrs.end();
  }
  PADDLE_ENFORCE_EQ(
      found, true,
      platform::errors::NotFound("(%s) is not found in AttributeMap.", name));
  return it->second;
}

template <typename VarType>
void BuildDygraphPtenKernelContext(
    const framework::KernelSignature& pt_kernel_signature,
    const pten::Kernel& pt_kernel, const NameVarMap<VarType>& ins,
    const NameVarMap<VarType>& outs, const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    platform::DeviceContext* dev_ctx, pten::KernelContext* kernel_ctx) {
  kernel_ctx->SetDeviceContext(dev_ctx);

  auto& input_names = std::get<0>(pt_kernel_signature.args);
  auto& attr_names = std::get<1>(pt_kernel_signature.args);
  auto& output_names = std::get<2>(pt_kernel_signature.args);

  auto& input_defs = pt_kernel.args_def().input_defs();
  auto& output_defs = pt_kernel.args_def().output_defs();
  auto& attr_defs = pt_kernel.args_def().attribute_defs();

  PADDLE_ENFORCE_EQ(input_names.size(), input_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of inputs_args names (%d) must be equal to "
                        "the size of kernel input_defs (%d).",
                        input_names.size(), input_defs.size()));

  PADDLE_ENFORCE_EQ(output_names.size(), output_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of outputs_args names (%d) must be equal to "
                        "the size of kernel output_defs (%d).",
                        output_names.size(), output_defs.size()));

  PADDLE_ENFORCE_EQ(attr_names.size(), attr_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of attribute_args names (%d) must be equal "
                        "to the size of kernel attribute_defs (%d).",
                        attr_names.size(), attr_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto& ins_vector = ins.at(input_names[i]);

    size_t start_idx = (i == 0 ? 0 : kernel_ctx->InputRangeAt(i - 1).second);
    size_t end_idx = start_idx + ins_vector.size();

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      const auto* tensor_in = GetTensorFromVar(ins_vector[offset]->Var());
      kernel_ctx->EmplaceBackInputWithoutSetRange(tensor_in);
    }
    kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
  }

  for (size_t i = 0; i < output_names.size(); ++i) {
    size_t start_idx = (i == 0 ? 0 : kernel_ctx->OutputRangeAt(i - 1).second);

    auto iter = outs.find(output_names[i]);
    if (iter == outs.end()) {
      kernel_ctx->EmplaceBackOutputWithoutSetRange({nullptr});
      kernel_ctx->AssignOutputRange(std::make_pair(start_idx, start_idx + 1),
                                    i);
      continue;
    }

    auto& outs_vector = iter->second;
    size_t end_idx = start_idx + outs_vector.size();

    for (size_t offset = 0; offset < outs_vector.size(); ++offset) {
      if (outs_vector[offset] == nullptr) {
        kernel_ctx->EmplaceBackOutputWithoutSetRange({nullptr});
        continue;
      }
      auto* var = outs_vector[offset]->MutableVar();
      framework::Tensor* tensor_out = nullptr;
      if (var->template IsType<framework::LoDTensor>()) {
        tensor_out = var->template GetMutable<framework::LoDTensor>();
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported output `%s` type when call pt kernel.",
            framework::ToTypeName(var->Type())));
      }  // TODO(zyfncg): Add support for SelectedRows

      experimental::ResetTensorByArgDef(tensor_out, output_defs.at(i));
      framework::SetAllocationForOutputTenosr(
          tensor_out, pten::TransToFluidPlace(output_defs.at(i).backend));

      kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
    }
    kernel_ctx->AssignOutputRange(std::make_pair(start_idx, end_idx), i);
  }

  for (size_t i = 0; i < attr_names.size(); ++i) {
    if (attr_defs[i].type_index == std::type_index(typeid(pten::ScalarArray))) {
      if (attrs.find(attr_names[i]) !=
          attrs.end()) {  // shape is in the attribute
        auto& attr = GetAttr(attrs, default_attrs, attr_names[i]);
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int64_t>))) {
          kernel_ctx->EmplaceBackAttr(std::move(
              pten::ScalarArray(BOOST_GET_CONST(std::vector<int64_t>, attr))));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(std::vector<int32_t>))) {
          kernel_ctx->EmplaceBackAttr(std::move(
              pten::ScalarArray(BOOST_GET_CONST(std::vector<int32_t>, attr))));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to VectorTensor when "
              "construct KernelContext.",
              attr_names[i]));
        }
      } else {  // shape is in the input
        auto& ins_vector = ins.at(attr_names[i]);
        if (ins_vector.size() == 1) {  // ShapeTensor
          kernel_ctx->EmplaceBackAttr(std::move(
              experimental::MakePtenScalarArrayFromVar(ins_vector[0]->Var())));
        } else {  // ShapeTensorList
          std::vector<framework::Variable*> variables;
          variables.reserve(ins_vector.size());
          for (const auto& var_base : ins_vector) {
            variables.push_back(var_base->MutableVar());
          }
          kernel_ctx->EmplaceBackAttr(std::move(
              experimental::MakePtenScalarArrayFromVarList(variables)));
        }
      }
    } else if (attr_defs[i].type_index ==
               std::type_index(typeid(pten::Scalar))) {
      // TODO(chenweihang): support other attrs later
      // TODO(zhangyunfei): Scalar should hold scaler type, and we should check
      // attribtue type by attr_defs
      if (attrs.find(attr_names[i]) != attrs.end() ||
          default_attrs.find(attr_names[i]) !=
              default_attrs.end()) {  // scalar is in the attribute
        auto& attr = GetAttr(attrs, default_attrs, attr_names[i]);
        if (std::type_index(attr.type()) == std::type_index(typeid(float))) {
          kernel_ctx->EmplaceBackAttr(
              std::move(pten::Scalar(BOOST_GET_CONST(float, attr))));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(std::string))) {
          kernel_ctx->EmplaceBackAttr(
              std::move(pten::Scalar(BOOST_GET_CONST(std::string, attr))));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(int))) {
          kernel_ctx->EmplaceBackAttr(
              std::move(pten::Scalar(BOOST_GET_CONST(int, attr))));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to Scalar when construct "
              "KernelContext in dygraph.",
              attr_names[i]));
        }
      } else {  // scalar is in the input
        auto& ins_vector = ins.at(attr_names[i]);
        kernel_ctx->EmplaceBackAttr(std::move(
            experimental::MakePtenScalarFromVar(ins_vector[0]->Var())));
      }

    } else {
      // TODO(chenweihang): support other attrs later
      auto& attr = GetAttr(attrs, default_attrs, attr_names[i]);
      if (attr_defs[i].type_index == std::type_index(typeid(int))) {
        kernel_ctx->EmplaceBackAttr(BOOST_GET_CONST(int, attr));
      } else if (attr_defs[i].type_index == std::type_index(typeid(float))) {
        kernel_ctx->EmplaceBackAttr(BOOST_GET_CONST(float, attr));
      } else if (attr_defs[i].type_index == std::type_index(typeid(bool))) {
        kernel_ctx->EmplaceBackAttr(BOOST_GET_CONST(bool, attr));
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(pten::DataType))) {
        auto data_type = pten::TransToPtenDataType(
            static_cast<framework::proto::VarType::Type>(
                BOOST_GET_CONST(int, attr)));
        kernel_ctx->EmplaceBackAttr(data_type);
      } else if (attr_defs[i].type_index ==
                 std::type_index(typeid(std::vector<int64_t>))) {
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int>))) {
          // Emplace Back Attr according to the type of Pten_Kernel args.
          const auto& vector_int_attr = BOOST_GET_CONST(std::vector<int>, attr);
          const std::vector<int64_t> vector_int64_attr(vector_int_attr.begin(),
                                                       vector_int_attr.end());
          kernel_ctx->EmplaceBackAttr(vector_int64_attr);
        }
        // TODO(YuanRisheng) Need support vector<int64_t> attr
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported cast op attribute `%s` when construct "
            "KernelContext in dygraph.",
            attr_names[i]));
      }
    }
  }
}

template <typename VarType>
void PreparePtenData(const pten::Kernel& pt_kernel,
                     const framework::KernelSignature& pt_kernel_signature,
                     const NameVarMap<VarType>& ins) {
  auto& input_names = std::get<0>(pt_kernel_signature.args);
  auto& input_defs = pt_kernel.args_def().input_defs();

  PADDLE_ENFORCE_EQ(input_names.size(), input_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of inputs_args names (%d) must be equal to "
                        "the size of kernel input_defs (%d).",
                        input_names.size(), input_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto& in_def = input_defs.at(i);
    auto& ins_vector = ins.at(input_names[i]);

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      auto var_base = ins_vector[offset];
      const auto* tensor_in = GetTensorFromVar(var_base->Var());
      if (tensor_in && tensor_in->IsInitialized()) {
        auto expected_place = pten::TransToFluidPlace(in_def.backend);
        if (platform::is_same_place(tensor_in->place(), expected_place)) {
          continue;
        }

        VLOG(3) << "Pten Transform Variable " << input_names[i] << " from "
                << tensor_in->place() << " to " << expected_place;

        framework::Tensor tmp_tensor;
        framework::TensorCopySync(*tensor_in, expected_place, &tmp_tensor);

        SetTensorToVariable(var_base->Var(), tmp_tensor,
                            var_base->MutableVar());
      }
    }
  }
}

}  // namespace imperative
}  // namespace paddle
