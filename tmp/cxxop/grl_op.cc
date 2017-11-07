#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("GrlOp")
    .Input("x: float")
    .Output("y: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
      })
    .SetIsStateful()
    .Doc(R"doc(
Gradient Reversal Layer proposed in paper 'Unsupervised Domain Adaptation by Backpropagation'.
)doc");

}  // end namespace tensorflow
