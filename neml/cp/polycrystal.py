from neml.math import nemlmath
from neml.cp import batch, singlecrystal
from neml.nlsolvers import newton

import numpy as np

class PolycrystalModel(object):
  """
    Parent class of polycrystal models.

    Basic interface is you pass in deformation and get out:
      1) Macro stress
      2) All history
      3) Macro tangent
      4) Macro energy
      5) Macro dissipation
      
    It makes available the micro:
      1) Deformation
      2) Stress
      3) History
      4) Orientation 
  """
  def __init__(self, model, orientations, T0 = 300.0):
    """
      Parameters:
        model:          crystal model
        orientations:   list of crystal orientations

      Keyword Args:
        T0:             initial temperature
    """
    self.model = model
    self.q0 = orientations
    self.T0 = T0

    self.N = len(self.q0)

    self.h_n = batch.init_history_batch(self.model, self.N)
    batch.set_orientation_passive_batch(self.model, self.h_n, self.q0)
    
    self.d_n = np.zeros((self.N,6))
    self.w_n = np.zeros((self.N,3))
    self.s_n = np.zeros((self.N,6))
    self.u_n = np.zeros((self.N,))
    self.p_n = np.zeros((self.N,))
    self.T_n = np.array([self.T0 for i in range(self.N)])
    
    self.L = np.zeros((3,3))
    self.s = np.zeros((6,))
    self.e = np.zeros((6,))
    self.T = self.T0
    self.t = 0.0
    self.u = 0.0
    self.p = 0.0

  def deformation_step(self, L, dt, T = 300.0, nthreads = 1):
    """
      Take a deformation step

      Parameters:
        L:           spatial velocity gradient
        dt:          time increment

      Keyword Args:
        T:           next temperature
        nthreads:    number of threads to use
    """
    # Advance the macrostep
    self.t += dt
    self.T = T
    self.L, self.s, A, B, T, self.u, self.p = self.take_step(L, dt, T, nthreads)

    self.e += nemlmath.sym(0.5*(self.L+self.L.T)) * dt

    # Accept the microstep
    self.d_n = np.copy(self.d_np1)
    self.w_n = np.copy(self.w_np1)
    self.s_n = np.copy(self.s_np1)
    self.u_n = np.copy(self.u_np1)
    self.p_n = np.copy(self.p_np1)
    self.T_n = np.copy(self.T_np1)
    self.h_n = np.copy(self.h_np1)

  def take_stress_step(self, sdir, erate, dt, T = 300.0, nthreads = 1, w = np.zeros((3,)),
      verbose = False, rtol = 1e-6, atol = 1e-8, miter = 20):
    """
      Take our usual stress-controlled step in a direction at a particular
      strain rate

      Parameters:
        sdir:       stress direction
        erate:      strain rate
        dt:         time increment

      Keyword Args:
        T:          temperature
        nthreads:   number of threads to use
        w:          macroscale spin
        verbose:    print verbose solver info
        rtol:       relative tolerance
        atol:       absolute tolerance
        miter:      maximum iterations
    """
    x0 = np.concatenate((sdir * erate * dt, [1,]))
    s_n = np.copy(self.s)

    def sfn(x):
      de = x[:6]
      a = x[6]

      L = nemlmath.usym(de) / dt + nemlmath.uskew(w)

      self.L, self.s, A, B, C, self.u, self.p = self.take_step(L, dt, T, nthreads)

      R = np.zeros((7,))
      J = np.zeros((7,7))

      R[:6] = self.s - s_n  - sdir * a
      R[6] = np.dot(sdir, de) / dt - erate

      J[:6,:6] = A
      J[:6,6] = -sdir
      J[6,:6] = sdir / dt
      J[6,6] = 0

      return R, J

    x = newton(sfn, x0, verbose = verbose, rtol = rtol, atol = atol, miter = miter)

    self.e += x[:6]

    # Accept the microstep
    self.T = T
    self.t += dt
    self.d_n = np.copy(self.d_np1)
    self.w_n = np.copy(self.w_np1)
    self.s_n = np.copy(self.s_np1)
    self.u_n = np.copy(self.u_np1)
    self.p_n = np.copy(self.p_np1)
    self.T_n = np.copy(self.T_np1)
    self.h_n = np.copy(self.h_np1) 

  def orientation(self, i):
    """
      Get the orientation for the ith crystal
      
      Parameters:
        i:       crystal number
    """
    return self.model.get_passive_orientation(self.h_n[i])

  @property
  def orientations(self):
    return [self.orientation(i) for i in range(self.N)]

class TaylorModel(PolycrystalModel):
  """
    Taylor homogenization where all crystals share the same deformation
  """
  def __init__(self, *args, **kwargs):
    """
      Parameters:
        model:           single crystal model
        orientations:    initial PASSIVE orientations

      Keyword Args:
        T0:              initial temperature
    """
    super().__init__(*args, **kwargs)

  def take_step(self, L, dt, T, nthreads = 1):
    """
      Actually take a deformation step and return the macroscale
      quantities.

      Store the microscale quantities in state n+1

      Parameters:
        L:           spatial velocity gradient
        dt:          time increment
        T:           temperature

      Keyword Args:
        nthreads:    number of threads in the update
    """
    d = nemlmath.sym(0.5*(L+L.T))
    w = nemlmath.skew(0.5*(L-L.T))

    self.d_np1 = np.array([d_n + d * dt for d_n in self.d_n])
    self.w_np1 = np.array([w_n + w * dt for w_n in self.w_n])
    self.T_np1 = np.array([T for i in range(self.N)])
    
    self.s_np1, self.h_np1, A_np1, B_np1, self.u_np1, self.p_np1 = batch.evaluate_crystal_batch(
        self.model, self.d_np1, self.d_n, self.w_np1, self.w_n, self.T_np1, self.T_n,
        self.t + dt, self.t, self.s_n, self.h_n, self.u_n, self.p_n,
        nthreads = nthreads)

    A_avg = np.mean(A_np1, axis = 0)
    B_avg = np.mean(B_np1, axis = 0)

    T = nemlmath.transform_fourth(A_avg, B_avg)

    return L, np.mean(self.s_np1, axis = 0), np.mean(A_np1, axis = 0), np.mean(B_np1, axis = 0), T, np.mean(self.u_np1), np.mean(self.p_np1)
