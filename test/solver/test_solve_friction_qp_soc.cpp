#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

// Mirrors the target application: a small QP (regularizing a contact force
// towards a desired value) subject to a friction-cone (SOC) constraint.
// With friction coefficient mu=1, the constraint set is exactly the
// standard Lorentz cone, so "min 0.5||f-f_des||^2 s.t. f in Q^3" is a
// Euclidean projection of f_des onto Q^3, which has a well-known closed
// form: for f_des=(v0,vbar) with a=||vbar||, if a<=v0 the projection is
// f_des itself; if a<=-v0 it is 0; otherwise it is
//   0.5*(v0+a) * (1, vbar/a).
TEST(SolveFrictionQpSoc, ProjectsDesiredForceOntoFrictionCone) {
  const Index n = 3;
  SparseMat P = testutil::makeSparse(n, n, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}});

  Vec f_des(3);
  f_des << 1.0, 3.0, 4.0;  // ||(3,4)|| = 5 > 1: desired force violates the cone
  Vec q = -f_des;

  // -f + s = 0, s in Q^3  =>  s = f, f in Q^3.
  SparseMat A = testutil::makeSparse(3, 3, {{0, 0, -1.0}, {1, 1, -1.0}, {2, 2, -1.0}});
  Vec b = Vec::Zero(3);

  ConeSpec spec;
  spec.soc_dims = {3};

  Solver solver;
  ASSERT_TRUE(solver.setup(P, q, A, b, spec));
  const Solution& sol = solver.solve();

  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);

  const Scalar v0 = f_des[0];
  const Scalar a = f_des.tail(2).norm();
  Vec expected(3);
  ASSERT_GT(a, v0);
  ASSERT_GT(a, -v0);
  expected[0] = 0.5 * (v0 + a);
  expected.tail(2) = (0.5 * (v0 + a) / a) * f_des.tail(2);

  testutil::expectVecNear(sol.x, expected, 1e-5);

  Mat Pd = Mat(Mat(P).selfadjointView<Eigen::Upper>());
  Mat Ad = Mat(A);
  testutil::expectKktOptimal(Pd, q, Ad, b, spec, sol, 1e-5);
}
