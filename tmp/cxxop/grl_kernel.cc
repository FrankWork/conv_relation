#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


namespace tensorflow {

class GrlOp : public OpKernel{
public:
  explicit GrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    
  }

  void Compute(OpKernelContext* ctx) override{
    // Grab the input tensor
    const Tensor& input_tensor = ctx->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = input(i);
    }

  }

};

REGISTER_KERNEL_BUILDER(Name("GrlOp").Device(DEVICE_CPU), GrlOp);

}  // end namespace tensorflow
