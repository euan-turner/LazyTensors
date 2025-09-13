#pragma once

#include "optim/optimiser.hpp"

namespace optim {

class SGD : public Optimiser {
  private:
    float lr_;

  public:
    explicit SGD(float lr) : lr_(lr) {}

    void step() override;
};

}