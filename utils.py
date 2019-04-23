import tensorflow as tf
from tensorflow.python.framework.function import Defun


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


class recomputable(object):
    '''A wrapper that allows us to choose whether we recompute
    activations during the backward pass, or we
    just use the function as normal.

    Usage:

    @recompute_option('func')
    def func(x, y):
        k = g(x, y)
        j = h(k)
        return j

    z1 = func(x, y, recompute=True)
    z2 = func(x, y, recompute=False)

    Behavior:
        z1 will not store activations for k and j, whereas z2 will.

    NOTE: args to `func` must be tensors. kwargs must not
    be tensors. kwargs must include recompute.

    IMPORTANT: variables should *not* be declared inside of this function,
    but rather be declared externally and then passed as tensor
    arguments!

    '''

    def __init__(self, name):
        self.name = name
        self.output_shape_cache = None
        self.normal_fn = None
        self.recompute_fns = {}

    def __call__(self, f):
        self.normal_fn = f
        return self.meta_fn

    def meta_fn(self, *args, **kwargs):
        # This function decides whether to build the recompute fn,
        # apply it, or use the non-recompute function.
        # It needs to build a new function for each new set of
        # kwargs.
        recompute = kwargs.pop('recompute')
        if not recompute:
            return self.normal_fn(*args, **kwargs)

        name = f"{self.name}"
        for key in kwargs:
            name += f"-{key}-{kwargs[key]}"
        try:
            size_hash = str(hash(int(
                ''.join([''.join([str(x) for x in a.shape.as_list()])
                         for a in args]))))[0:6]
        except AttributeError:
            raise ValueError('Non-tensor arguments must be keyword arguments.')
        name += size_hash
        if name not in self.recompute_fns:
            print('INFO: Defining function:', name)
            self.recompute_fns[name] = self.build_fns(name, args, kwargs)
        return self.recompute_fns[name](*args)

    def build_fns(self, name, outer_args, outer_kwargs):
        input_shapes = [x.get_shape() for x in outer_args]
        output_shape_cache = None

        @Defun(func_name=name + "_bwd", noinline=False)
        def bwd(*args):
            nonlocal output_shape_cache
            nonlocal input_shapes
            fwd_args = args[:-1]
            dy = args[-1]
            for i, a in enumerate(fwd_args):
                a.set_shape(input_shapes[i])
            with tf.device("/gpu:0"), tf.control_dependencies([dy]):
                y = self.normal_fn(*fwd_args, **outer_kwargs)
            gs = tf.gradients(ys=[y], xs=fwd_args, grad_ys=[dy])
            return gs

        @Defun(func_name=name, noinline=False, grad_func=bwd,
               shape_func=lambda x: output_shape_cache)
        def fwd(*args):
            nonlocal output_shape_cache
            nonlocal input_shapes
            with tf.device("/gpu:0"):
                fwd_args = args
                for i, a in enumerate(args):
                    a.set_shape(input_shapes[i])
                y = self.normal_fn(*fwd_args, **outer_kwargs)
            if not output_shape_cache:
                try:
                    output_shape_cache = [o.get_shape() for o in y]
                except TypeError:
                    output_shape_cache = [y.get_shape()]
            return y

        return fwd
