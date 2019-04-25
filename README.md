**Status:** Archive (code is provided as-is, no updates expected)

# Sparse Attention

This repository contains the sparse attention primitives used in Sparse Transformers (see [blog](https://openai.com/blog/sparse-transformer) and [paper](https://arxiv.org/abs/1904.10509)). Specifically, it includes the following:

1) A faster implementation of normal attention (the upper triangle is not computed, and many operations are fused).
2) An implementation of "strided" and "fixed" attention, as in the Sparse Transformers paper.
3) A simple recompute decorator, which can be adapted for usage with attention.

We hope this code can further accelerate research into sparse attention.

An example Transformer implementation which is close to the version we use internally can be found at https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py. 

# Overview of kernels
The repository contains fused implementations of the attention operation, which takes in `Q`, `K`, `V` matrices (all of dimensionality `batch, time, dim`) representing the queries, keys, and values for a sequence. For every query element, a weighted sum of the values is returned, where the weightings are determined by the scaled matrix product of `Q` and `K^T`.

The kernels allow specification of block sparsity in the `QK^T` matrix. This means you define a pattern of 0/1s on a `[time/blocksize, time/blocksize]` matrix of blocks, and the values where it is 0 will not be computed, and not be included in the softmax calculation. Additionally, one can define "callbacks" on the computed blocks, which will further mask out values in any given block from the softmax (though the matrix product will still be computed for those elements). 

Block sizes of `{8, 16, 32, 64}` are supported, and slight advantages in speed may be seen from using larger blocks.

# Prerequisites
For fp32 and blocksize `32`, any NVIDIA GPU past Kepler can be used (i.e. compute capability beyond 3.5).

For fp16 and blocksize `8, 16, 32, 64`, a GPU with Tensor Cores (e.g. the V100 GPU, compute capability >= 7.0) is required.

The primary dependency is the OpenAI [blocksparse](https://github.com/openai/blocksparse/) package.

With CUDA 10 and tensorflow-gpu, you can install blocksparse with `pip install blocksparse`.

For other setups, you must install blocksparse from source, and directions can be found in the [root of the repository](https://github.com/openai/blocksparse/).

# Examples

Run the following on a non-V100 GPU:
```
python attention.py
```

On a V100 GPU:
```
python attention.py fp16
```

# General usage
An example can be found at the bottom of `attention.py`.

```python

full_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="all", recompute=True)
full_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="all", recompute=True)

# first step of strided attention
local_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=32, recompute=True)
local_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=32, recompute=True)

# second step of strided attention
strided_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=32, recompute=True)
strided_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=32, recompute=True)

# # the 'fixed' attention pattern
fixed = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="fixed", local_attn_ctx=128, num_verts=4, vertsize=1, recompute=True)

```

# Referencing this work

If you find this helpful in your work, you can consider citing the following:

```
@article{child2019sparsetransformer,
  title={Generating Long Sequences with Sparse Transformers},
  author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  journal={URL https://openai.com/blog/sparse-transformers},
  year={2019}
}
```
