/* Copyright 2019-2024 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "nncase/ntt/vector.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;


TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C, H, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C, H, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C * P, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 4, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C, H * P, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C * P, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 4, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C * P, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<C, H * P, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<C, H * P, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C, H, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C, H, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_0_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C * P, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 4, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C, H * P, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_0_1_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C * P, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 3, 1, 4, 2};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C * P, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_1_2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t C = 1;
    constexpr size_t H = 77;
    constexpr size_t W = 3;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(C, H * P, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 3, 2, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(C, H * P, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_add5_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_add5_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_add5_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_add5_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_mul2_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_mul2_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_mul2_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim2_mul2_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_add5_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_add5_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_add5_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_add5_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim2_add5_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim2_add5_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim2_add5_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, (H) +7, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim2_mul2_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim2_mul2_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim2_mul2_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, (H) *2, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) *2, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_add5_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_add5_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_add5_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) +7, H, W>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_add5_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_add5_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_add5_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_add5_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_mul2_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_mul2_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_mul2_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_mul2_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_add5_unpack_axis_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_add5_unpack_axis_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_add5_unpack_axis_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_add5_unpack_axis_0_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, H, W));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim2_add5_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim2_add5_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim2_add5_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, (H) +7, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim2_mul2_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim2_mul2_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim2_mul2_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, (H) *2, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) *2, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_add5_unpack_axis_0_1_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C * P, H, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 4, 1, 5, 2, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_add5_unpack_axis_1_2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H * P, W));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 4, 2, 5, 3};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_add5_unpack_axis_2_3_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) +7, H, W));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 4, 3, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W, D * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W, D * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W, D * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_contiguous_unpack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C, H, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W, D * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W, D * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W, D * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_unpack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C, H, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C * P, H, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 6, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H * P, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 6, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W * P, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 6, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_contiguous_unpack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P, D * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N * P, C * P, H, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 6, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N * P, C * P, H, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C * P, H * P, W, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 6, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C * P, H * P, W, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H * P, W * P, D>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P, D>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 6, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H * P, W * P, D>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_fixed_2D_vector_non_contiguous_dim1_mul2_unpack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::fixed_shape_v<N, (C) *2, H, W, D>);
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<N, C, H, W, D>,
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<N, C, H, W * P, D * P>);
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D * P>);
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::fixed_shape_v<N, C, H, W, D>);
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<N, C, H, W * P, D * P>);
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W, D * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W, D * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W, D * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_contiguous_unpack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C, H, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W, D * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W, D * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 4, 5};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W, D * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_unpack_axis_0_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C, H, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C * P, H, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 6, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H * P, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 6, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W * P, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 6, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_contiguous_unpack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, C, H, W, D));
    NttTest::init_tensor(ntt_input, min_input, max_input);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P, D * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_0_1_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N * P, C * P, H, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<0, 1>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 5, 1, 6, 2, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N * P, C * P, H, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N * P, C * P, H, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_1_2_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C * P, H * P, W, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<1, 2>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 5, 2, 6, 3, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C * P, H * P, W, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C * P, H * P, W, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_2_3_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H * P, W * P, D));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<2, 3>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P, D));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 5, 3, 6, 4};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H * P, W * P, D};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H * P, W * P, D));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
TEST(UnpackTest_Float8e5m2, Float8e5m2_dynamic_2D_vector_non_contiguous_dim1_mul2_unpack_axis_3_4_5D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    constexpr size_t N = 2;
    constexpr size_t C = 8;
    constexpr size_t H = 4;
    constexpr size_t W = 4;
    constexpr size_t D = 2;
    float_e5m2_t min_input = float_e5m2_t(-57344.0f);
    float_e5m2_t max_input = float_e5m2_t(57344.0f);

    // ------------------------------------------------------------------
    // 1. create NTT input 
    // ------------------------------------------------------------------
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P, P>>(ntt::make_shape(N, (C) *2, H, W, D));
    NttTest::init_tensor(big_tensor, min_input, max_input);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(N, C, H, W, D),
        big_tensor.strides());
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(N, C, H, W * P, D * P));
    
    // Execute unpack operation
    ntt::unpack(ntt_input, ntt_output1, ntt::fixed_shape_v<3, 4>);
    
    auto ntt_output1_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D * P));
    NttTest::reinterpret_cast_fp8_to_uint8(ntt_output1, ntt_output1_uint8);
    
    
    // ------------------------------------------------------------------
    // 1. build ORT input tensor
    // ------------------------------------------------------------------
        auto ntt_input_uint8 = ntt::make_tensor<ntt::vector<uint8_t, P, P>>(ntt::make_shape(N, C, H, W, D));
        NttTest::reinterpret_cast_fp8_to_uint8(ntt_input, ntt_input_uint8);
    
        auto ort_input = NttTest::ntt2ort(ntt_input_uint8);
    
    // ------------------------------------------------------------------
    // 2. call ortki kernel to generate ORT output
    // ------------------------------------------------------------------
    // ORT reference implementation (kernel part)
    int64_t perms[] = {0, 1, 2, 3, 5, 4, 6};
    auto transposed_tensor = ortki_Transpose(ort_input, perms, std::size(perms));
    int64_t reshape_data[] = {N, C, H, W * P, D * P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_output = ortki_Reshape(transposed_tensor, shape_tensor, 0);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden_uint8 = ntt::make_tensor<uint8_t>(ntt::make_shape(N, C, H, W * P, D * P));
    NttTest::ort2ntt(ort_output, ntt_golden_uint8);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1_uint8, ntt_golden_uint8));
    }
    
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
