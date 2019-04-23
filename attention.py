import sys
import numpy as np
import tensorflow as tf
from blocksparse import BlocksparseTransformer
from utils import shape_list, recomputable


def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = tf.minimum(n - 1, bandwidth - 1)
        b = tf.matrix_band_part(tf.ones([n, n]), ctx, 0)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = tf.reshape(tf.range(n, dtype=tf.int32), [n, 1])
        y = tf.transpose(x)
        z = tf.zeros([n, n], dtype=tf.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = tf.equal(tf.floormod(q - k, stride), 0)
        c3 = tf.logical_and(c1, c2)
        b = tf.cast(c3, tf.float32)
    else:
        raise ValueError('Not yet implemented')
    b = tf.reshape(b, [1, 1, n, n])
    return b


def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = shape_list(x)
    x = tf.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [n, t, embd])
    return x


def split_heads(x, n):
    return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)


def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


@recomputable('attention_impl')
def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = shape_list(k)[2]
    mask = tf.to_float(get_attn_mask(n_timesteps, attn_mode, local_attn_ctx))
    w = tf.matmul(q, k, transpose_b=True)
    scale_amount = 1.0 / np.sqrt(shape_list(q)[-1])
    orig_dtype = q.dtype
    if orig_dtype == tf.float16:
        w = tf.cast(w, tf.float32)
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = tf.nn.softmax(w)
    w = tf.cast(w, orig_dtype)
    a = tf.matmul(w, v)
    a = merge_heads(a)
    return a


@recomputable('blocksparse_attention_impl')
def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None,
                               blocksize=32, num_verts=None, vertsize=None):
    n_ctx = shape_list(q)[1]
    if attn_mode == 'strided':
        # Strided attention is implemented on the transposed matrix to provide greater block sparsity
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = shape_list(q)[-1] // heads
    bst = get_blocksparse_obj(n_ctx, heads, attn_mode, blocksize, local_attn_ctx, num_verts, vertsize)
    scale_amount = tf.cast(1.0 / np.sqrt(n_state), tf.float32)
    w = bst.query_key_op(q, k)
    w = bst.masked_softmax(w, scale=scale_amount)
    a = bst.weight_value_op(w, v)
    if attn_mode == 'strided':
        n, t, embd = shape_list(a)
        bT_ctx = n_ctx // local_attn_ctx
        a = tf.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = tf.transpose(a, [0, 2, 1, 3])
        a = tf.reshape(a, [n, t, embd])
    return a


def get_blocksparse_obj(n_ctx, n_heads, attn_mode, blocksize=32, local_attn_ctx=None, num_verts=4, vertsize=1):
    '''Defines the block-level sparsity pattern in the attention matrix. Enabled blocks
    will have the callback called on them in order to define a positionwise sparsity mask.'''
    n_bctx = n_ctx // blocksize
    layout = np.ones([n_bctx, n_bctx], dtype=np.bool)
    extra_diagonals = None
    block_chunks = None

    if attn_mode in ['all', 'fixed']:
        pass
    elif attn_mode == 'local':
        assert local_attn_ctx % blocksize == 0
        extra_diagonals = local_attn_ctx // blocksize
    elif attn_mode == 'strided':
        bT_ctx = n_ctx // local_attn_ctx
        assert bT_ctx % blocksize == 0
        block_chunks = bT_ctx // blocksize
    else:
        raise ValueError(f'attn mode {attn_mode} invalid')

    if attn_mode == 'fixed':
        assert n_heads % num_verts == 0
        lctx = local_attn_ctx
        stride = lctx // blocksize
        assert vertsize <= stride
        assert stride % vertsize == 0
        indices = [i for i in range(stride - 1, -1, -1)]
        indices = np.array(indices).reshape([-1, vertsize])
        if num_verts == 1:
            layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
            for idx in indices[0]:
                layout[:, idx::stride] = 1
            for q_idx in range(n_bctx):
                # Each thing can attend to its local block
                row = q_idx // stride
                layout[q_idx, row * stride:(row + 1) * stride] = 1
                # Any query cannot attend to keys above it
                layout[q_idx, q_idx + 1:] = 0
        else:
            layouts = []
            indices = indices[:num_verts]
            for h in range(n_heads):
                layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
                subindices = indices[h % num_verts]
                for idx in subindices:
                    layout[:, idx::stride] = 1
                for q_idx in range(n_bctx):
                    # Each position can attend to its local block
                    row = q_idx // stride
                    layout[q_idx, row * stride:(row + 1) * stride] = 1
                    # Any query cannot attend to keys above it
                    layout[q_idx, q_idx + 1:] = 0
                layouts.append(layout)
            layout = np.array(layouts)
    else:
        for q_idx, k_idx in np.ndindex(n_bctx, n_bctx):
            if k_idx > q_idx:
                layout[q_idx, k_idx] = 0
            if extra_diagonals and k_idx + extra_diagonals < q_idx:
                layout[q_idx, k_idx] = 0
            if block_chunks is not None:
                layout[q_idx, k_idx] = 0
                offset = q_idx % block_chunks
                if k_idx + offset >= q_idx and k_idx <= q_idx:
                    layout[q_idx, k_idx] = 1
    bst = BlocksparseTransformer(layout, block_size=blocksize,
                                 mask_callback=get_callback(attn_mode, local_attn_ctx),
                                 heads=n_heads)
    return bst


def get_callback(attn_mode, local_attn_ctx=None):
    '''Defines a function which returns the positionwise sparsity pattern for every block
    that is enabled in the blocksparse object
    '''
    def cb(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
        mask = np.ones(blk_shape, dtype=np.bool)

        # on the diagonal blocks mask out the upper diagonal
        if qry_idx == key_idx:
            for q, k in np.ndindex(blk_shape):
                if k > q:
                    mask[q, k] = 0
        if attn_mode in ['all', 'strided', 'fixed']:
            return mask
        if attn_mode == 'local':
            bandwidth = local_attn_ctx
            # convert group indices to absolute indices and mask
            # according to that
            q_pos = blk_shape[0] * qry_idx
            k_pos = blk_shape[1] * key_idx
            for q, k in np.ndindex(blk_shape):
                q_ = q + q_pos
                k_ = k + k_pos
                if k_ > q_ or k_ + bandwidth <= q_:
                    mask[q, k] = 0
            return mask
        raise ValueError
    return cb


if __name__ == '__main__':
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    is_fp16 = len(sys.argv) > 1 and sys.argv[1] == 'fp16'

    dtype = tf.float16 if is_fp16 else tf.float32
    blocksize = 32
    # query, key, values should be batch x time x dim.
    q = tf.random_normal(shape=[4, 1024, 256], dtype=dtype)
    k = tf.random_normal(shape=[4, 1024, 256], dtype=dtype)
    v = tf.random_normal(shape=[4, 1024, 256], dtype=dtype)

    full_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="all", recompute=True)
    full_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="all", blocksize=blocksize, recompute=True)

    # # first step of strided attention
    local_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=32, blocksize=blocksize, recompute=True)
    local_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="local", local_attn_ctx=32, recompute=True)

    # # second step of strided attention
    strided_attn_bs = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=32, blocksize=blocksize, recompute=True)
    strided_attn_tf = attention_impl(q, k, v, heads=4, attn_mode="strided", local_attn_ctx=32, recompute=True)

    # # # the 'fixed' attention pattern
    fixed = blocksparse_attention_impl(q, k, v, heads=4, attn_mode="fixed", local_attn_ctx=128, num_verts=4, vertsize=1, blocksize=blocksize, recompute=True)
    sess = tf.Session()

    fatf, fabs, latf, labs, satf, sabs, fixed_bs = sess.run([
        full_attn_tf, full_attn_bs, local_attn_tf, local_attn_bs, strided_attn_tf, strided_attn_bs, fixed])

    print(fatf[0])
    print(fabs[0])
    print('-----')
    print(latf[0])
    print(labs[0])
    print('-----')
    print(satf[0])
    print(sabs[0])
    print('-----')
    print(fixed_bs[0])
