#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicDevice, SymbolicSize
#include <sgl_kernel/type.cuh>  // For device::cast, bf16_t, fp32_t
#include <sgl_kernel/utils.h>   // For RuntimeCheck
#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace rwkv7_backend::jit {

constexpr float kWScale = -0.6065306597126334f;

template <int kHeadDim>
struct RWKV7PrefillParams {
  const void* __restrict__ r;
  const void* __restrict__ w_logits;
  const void* __restrict__ k;
  const void* __restrict__ v;
  const void* __restrict__ kk;
  const void* __restrict__ a;
  const int64_t* __restrict__ seq_indptr;
  const float* __restrict__ initial_state;
  void* __restrict__ output;
  float* __restrict__ final_state;
  uint32_t num_heads;
};

template <int kHeadDim>
struct RWKV7DecodeParams {
  const void* __restrict__ r;
  const void* __restrict__ w_logits;
  const void* __restrict__ k;
  const void* __restrict__ v;
  const void* __restrict__ kk;
  const void* __restrict__ a;
  const float* __restrict__ initial_state;
  void* __restrict__ output;
  float* __restrict__ final_state;
  uint32_t num_heads;
};

template <typename DType>
SGL_DEVICE float load_float(const void* ptr, int64_t idx) {
  return device::cast<fp32_t>(static_cast<const DType*>(ptr)[idx]);
}

template <typename DType>
SGL_DEVICE void store_output(void* ptr, int64_t idx, float value) {
  static_cast<DType*>(ptr)[idx] = device::cast<DType>(value);
}

template <int kHeadDim, typename DType>
__global__ void rwkv7_prefill_kernel(const RWKV7PrefillParams<kHeadDim> params) {
  const uint32_t head_idx = blockIdx.x;
  const uint32_t seq_idx = blockIdx.y;
  const uint32_t v_idx = threadIdx.x;

  static_assert(kHeadDim > 0);
  if (v_idx >= kHeadDim) {
    return;
  }

  const int64_t seq_start = params.seq_indptr[seq_idx];
  const int64_t seq_end = params.seq_indptr[seq_idx + 1];
  const int64_t state_base =
      (static_cast<int64_t>(seq_idx) * params.num_heads + head_idx) * kHeadDim * kHeadDim;

  float state[kHeadDim];
  for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
    state[k_idx] = params.initial_state[state_base + static_cast<int64_t>(k_idx) * kHeadDim + v_idx];
  }

  __shared__ float r_shared[kHeadDim];
  __shared__ float decay_shared[kHeadDim];
  __shared__ float k_shared[kHeadDim];
  __shared__ float kk_shared[kHeadDim];
  __shared__ float a_shared[kHeadDim];

  for (int64_t token_idx = seq_start; token_idx < seq_end; ++token_idx) {
    const int64_t token_base = (token_idx * params.num_heads + head_idx) * kHeadDim;

    __syncthreads();
    r_shared[v_idx] = load_float<DType>(params.r, token_base + v_idx);
    const float w_logit = load_float<DType>(params.w_logits, token_base + v_idx);
    decay_shared[v_idx] = __expf(kWScale / (1.0f + __expf(-w_logit)));
    k_shared[v_idx] = load_float<DType>(params.k, token_base + v_idx);
    kk_shared[v_idx] = load_float<DType>(params.kk, token_base + v_idx);
    a_shared[v_idx] = load_float<DType>(params.a, token_base + v_idx);
    const float v_value = load_float<DType>(params.v, token_base + v_idx);
    __syncthreads();

    float recurrence = 0.0f;
    for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
      recurrence += state[k_idx] * kk_shared[k_idx];
    }
    recurrence = -recurrence;

    float y_value = 0.0f;
    for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
      float next_state = state[k_idx];
      next_state = next_state * decay_shared[k_idx];
      next_state += recurrence * kk_shared[k_idx] * a_shared[k_idx];
      next_state += k_shared[k_idx] * v_value;
      y_value += next_state * r_shared[k_idx];
      state[k_idx] = next_state;
    }

    store_output<DType>(params.output, token_base + v_idx, y_value);
  }

  for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
    params.final_state[state_base + static_cast<int64_t>(k_idx) * kHeadDim + v_idx] = state[k_idx];
  }
}

template <int kHeadDim, typename DType>
__global__ void rwkv7_decode_kernel(const RWKV7DecodeParams<kHeadDim> params) {
  const uint32_t batch_idx = blockIdx.y;
  const uint32_t head_idx = blockIdx.x;
  const uint32_t v_idx = threadIdx.x;

  static_assert(kHeadDim > 0);
  if (v_idx >= kHeadDim) {
    return;
  }

  const int64_t token_base = (static_cast<int64_t>(batch_idx) * params.num_heads + head_idx) * kHeadDim;
  const int64_t state_base =
      (static_cast<int64_t>(batch_idx) * params.num_heads + head_idx) * kHeadDim * kHeadDim;

  float state[kHeadDim];
  for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
    state[k_idx] = params.initial_state[state_base + static_cast<int64_t>(k_idx) * kHeadDim + v_idx];
  }

  __shared__ float r_shared[kHeadDim];
  __shared__ float decay_shared[kHeadDim];
  __shared__ float k_shared[kHeadDim];
  __shared__ float kk_shared[kHeadDim];
  __shared__ float a_shared[kHeadDim];

  __syncthreads();
  r_shared[v_idx] = load_float<DType>(params.r, token_base + v_idx);
  const float w_logit = load_float<DType>(params.w_logits, token_base + v_idx);
  decay_shared[v_idx] = __expf(kWScale / (1.0f + __expf(-w_logit)));
  k_shared[v_idx] = load_float<DType>(params.k, token_base + v_idx);
  kk_shared[v_idx] = load_float<DType>(params.kk, token_base + v_idx);
  a_shared[v_idx] = load_float<DType>(params.a, token_base + v_idx);
  const float v_value = load_float<DType>(params.v, token_base + v_idx);
  __syncthreads();

  float recurrence = 0.0f;
  for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
    recurrence += state[k_idx] * kk_shared[k_idx];
  }
  recurrence = -recurrence;

  float y_value = 0.0f;
  for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
    float next_state = state[k_idx];
    next_state = next_state * decay_shared[k_idx];
    next_state += recurrence * kk_shared[k_idx] * a_shared[k_idx];
    next_state += k_shared[k_idx] * v_value;
    y_value += next_state * r_shared[k_idx];
    state[k_idx] = next_state;
  }

  store_output<DType>(params.output, token_base + v_idx, y_value);
  for (int k_idx = 0; k_idx < kHeadDim; ++k_idx) {
    params.final_state[state_base + static_cast<int64_t>(k_idx) * kHeadDim + v_idx] = state[k_idx];
  }
}

template <int kHeadDim, typename DType>
struct RWKV7PrefillKernel {
  static_assert(kHeadDim <= 128, "RWKV7 JIT prefill only supports head_dim <= 128");

  static void
  run(const tvm::ffi::TensorView r,
      const tvm::ffi::TensorView w_logits,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView kk,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView seq_indptr,
      const tvm::ffi::TensorView initial_state,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView final_state) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto T = SymbolicSize{"sequence_length"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto N = SymbolicSize{"num_sequences"};
    auto N1 = SymbolicSize{"num_sequences_plus_one"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    D.set_value(kHeadDim);

    TensorMatcher({B, T, H, D}).with_dtype<DType>().with_device(device).verify(r).verify(w_logits).verify(k).verify(v).verify(kk).verify(a).verify(output);
    TensorMatcher({N1}).with_dtype<int64_t>().with_device(device).verify(seq_indptr);
    TensorMatcher({N, H, D, D}).with_dtype<fp32_t>().with_device(device).verify(initial_state).verify(final_state);

    RuntimeCheck(N1.unwrap() > 1, "seq_indptr must contain at least two entries");
    RuntimeCheck(
        N1.unwrap() == N.unwrap() + 1,
        "seq_indptr length must equal num_sequences + 1, got ",
        N1.unwrap(),
        " vs ",
        N.unwrap());

    const auto num_sequences_plus_one = static_cast<uint32_t>(N1.unwrap());
    const auto num_sequences = num_sequences_plus_one - 1;

    const auto params = RWKV7PrefillParams<kHeadDim>{
        .r = r.data_ptr(),
        .w_logits = w_logits.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .kk = kk.data_ptr(),
        .a = a.data_ptr(),
        .seq_indptr = static_cast<const int64_t*>(seq_indptr.data_ptr()),
        .initial_state = static_cast<const float*>(initial_state.data_ptr()),
        .output = output.data_ptr(),
        .final_state = static_cast<float*>(final_state.data_ptr()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
    };

    LaunchKernel(dim3(params.num_heads, num_sequences), dim3(kHeadDim), device.unwrap())(
        rwkv7_prefill_kernel<kHeadDim, DType>,
        params);
  }
};

template <int kHeadDim, typename DType>
struct RWKV7DecodeKernel {
  static_assert(kHeadDim <= 128, "RWKV7 JIT decode only supports head_dim <= 128");

  static void
  run(const tvm::ffi::TensorView r,
      const tvm::ffi::TensorView w_logits,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView kk,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView initial_state,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView final_state) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto T = SymbolicSize{"sequence_length"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    D.set_value(kHeadDim);

    TensorMatcher({B, T, H, D}).with_dtype<DType>().with_device(device).verify(r).verify(w_logits).verify(k).verify(v).verify(kk).verify(a).verify(output);
    TensorMatcher({B, H, D, D}).with_dtype<fp32_t>().with_device(device).verify(initial_state).verify(final_state);

    RuntimeCheck(T.unwrap() == 1, "RWKV7 decode JIT expects sequence_length == 1, got ", T.unwrap());

    const auto params = RWKV7DecodeParams<kHeadDim>{
        .r = r.data_ptr(),
        .w_logits = w_logits.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .kk = kk.data_ptr(),
        .a = a.data_ptr(),
        .initial_state = static_cast<const float*>(initial_state.data_ptr()),
        .output = output.data_ptr(),
        .final_state = static_cast<float*>(final_state.data_ptr()),
        .num_heads = static_cast<uint32_t>(H.unwrap()),
    };

    LaunchKernel(dim3(params.num_heads, static_cast<uint32_t>(B.unwrap())), dim3(kHeadDim), device.unwrap())(
        rwkv7_decode_kernel<kHeadDim, DType>,
        params);
  }
};

}  // namespace rwkv7_backend::jit
