# ConicXX

**ConicXX** is a modern C++ solver for quadratic conic optimization problems.

It implements a primal-dual interior-point method with:
- Nesterov–Todd scaling
- Mehrotra predictor–corrector
- Support for nonnegative and second-order cones
- Symmetric KKT systems with regularization

Built on Eigen, ConicXX is designed for research, prototyping, and high-performance applications.

## Literature

- Clarabel: https://arxiv.org/abs/2405.12762
- QOCO: https://arxiv.org/abs/2503.12658
- IPM slides with self-dual homogeneous embedding (as in Clarabel): https://www.syscop.de/files/2015ss/numopt/TEMPO_NOC_ECOS.pdf