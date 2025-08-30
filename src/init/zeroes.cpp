#include "init/zeroes.hpp"

namespace init {

void Zeroes::initialise(Tensor& tensor) {
  std::vector<size_t> dims = tensor.dims();

  std::vector<size_t> idx(dims.size(), 0);

  // TODO: Move some of this into TensorImpl so it can be made more efficient
  for (size_t flat = 0; flat < tensor.numel(); ++flat) {
    size_t rem = flat;
    for (size_t i = 0; i < tensor.dims().size(); ++i) {
      idx[i] = rem / tensor.strides()[i];
      rem %= tensor.strides()[i];
    }
    tensor.set(idx, 0.0f);
  }
}

}