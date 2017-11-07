from tensorflow.python.framework import ops

@ops.RegisterGradient("GrlOp")
def _grl_grad(op, grad):
  """The gradients for `grl`.

  Args:
    op: The `grl` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `grl` op.

  Returns:
    Gradients with respect to the input of `grl`.
  """
  return [-grad]  # List of one Tensor, since we have one input
