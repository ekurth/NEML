#!/usr/bin/env python3

import sys
sys.path.append('..')

from neml import solvers, models, elasticity, drivers, surfaces, hardening, visco_flow, general_flow

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  E = 151000.0
  nu = 0.33

  n = 12.0
  eta = 150.0
  k = 6.0
  C = 24800.0
  g0 = 300.0
  Q = 86 - k
  gs = 300.0
  b = 5.0
  beta = 0.0

  erate = 1.0e-4
  emax = 0.75
  nsteps = 500
  
  elastic = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")

  surface = surfaces.IsoKinJ2()
  iso = hardening.VoceIsotropicHardeningRule(k, Q, b)
  cs = [C]
  gs = [hardening.SatGamma(gs, g0, beta)]
  As = [0.0]
  ns = [1.0]
  hmodel = hardening.Chaboche(iso, cs, gs, As, ns)

  fluidity = visco_flow.ConstantFluidity(eta)

  vmodel = visco_flow.ChabocheFlowRule(
      surface, hmodel, fluidity, n)

  flow = general_flow.TVPFlowRule(elastic, vmodel)

  model_treusdell = models.GeneralIntegrator(elastic, flow)
  model_jaumann = models.GeneralIntegrator(elastic, flow, truesdell = False)

  res1 = drivers.uniaxial_test(model_treusdell, erate, emax = emax, nsteps = nsteps)
  res2 = drivers.uniaxial_test(model_treusdell, erate, emax = emax, nsteps = nsteps, large_kinematics = True)

  res3 = drivers.uniaxial_test(model_jaumann, erate, emax = emax, nsteps = nsteps)
  res4 = drivers.uniaxial_test(model_jaumann, erate, emax = emax, nsteps = nsteps, large_kinematics = True)

  plt.plot(res1['strain'], res1['stress'], label = "Truesdell, small deformation")
  plt.plot(res2['strain'], res2['stress'], label = "Truesdell, large deformation")
  plt.plot(res3['strain'], res3['stress'], label = "Jaumann, small deformation")
  plt.plot(res4['strain'], res4['stress'], label = "Jaumann, large deformation")
  plt.legend(loc='best')
  plt.xlabel("Strain")
  plt.ylabel("Stress")
  plt.show()
