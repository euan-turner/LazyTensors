#include "init/xavier.hpp"

namespace init {

Xavier::Xavier(size_t fan_in, size_t fan_out, float gain, unsigned seed) 
  : _fan_in(fan_in), _fan_out(fan_out), _gain(gain) {
  _gen = std::mt19937(seed);
}

void Xavier::initialise(Tensor& tensor) {
  float std = _gain * std::sqrt(2.0f / (_fan_in + _fan_out));
  std::normal_distribution<float> dist(0.0f, std);

  std::vector<size_t> dims = tensor.dims();

  std::vector<size_t> idx(dims.size(), 0);

  // TODO: Move some of this into TensorImpl so it can be made more efficient
  for (size_t flat = 0; flat < tensor.numel(); ++flat) {
    size_t rem = flat;
    for (size_t i = 0; i < tensor.dims().size(); ++i) {
      idx[i] = rem / tensor.strides()[i];
      rem %= tensor.strides()[i];
    }
    tensor.set(idx, dist(_gen));
  }
}

}