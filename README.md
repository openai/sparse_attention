**Status:** Archive (code is provided as-is, no updates expected)

# Sparse Attention

This repository contains the sparse attention primitives used in [Sparse Transformers](https://openai.com/blog/sparse-transformer). Specifically, it includes the following:

1) A faster implementation of normal attention (the upper triangle is not computed, and many operations are fused).
2) An implementation of "strided" and "fixed" attention, as in the Sparse Transformers paper.
3) A simple recompute decorator, which can be adapted for usage with attention.

We hope this code can further accelerate research into sparse attention.

# Prerequisites
A GPU with Tensor Cores (e.g. the V100 GPU) is required.

The primary dependency is the OpenAI [blocksparse](https://github.com/openai/blocksparse/) package.

With CUDA 10 and tensorflow-gpu, you can install blocksparse with `pip install blocksparse`.

For other setups, you must install blocksparse from source, and directions can be found in the [root of the repository](https://github.com/openai/blocksparse/).

# Example usage
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

If you find this helpful in your work, consider citing the following:

```
@article{child2019sparsetransformer,
  title={Generating Long Sequences with Sparse Transformers},
  author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  journal={URL https://openai.com/blog/sparse-transformers},
  year={2019}
}
```
