Rate independent plasticity
===========================

Overview
--------

This class implements rate independent plasticity described by:

   The elastic trial state:

   .. math::

      \bm{\varepsilon}^{p}_{tr} = \bm{\varepsilon}^{p}_n

      \bm{\sigma}_{tr} = \mathbf{\mathfrak{C}}_{n+1} : 
         \left( \bm{\varepsilon}_{n+1} - \bm{\varepsilon}_{tr}^p  \right)

      \bm{\alpha}_{tr} = \bm{\alpha}_{n}

   The plastic correction:

   .. math::
      \bm{\sigma}_{n+1} = \mathbf{\mathfrak{C}}_{n+1} : 
         \left( \bm{\varepsilon}_{n+1} - \bm{\varepsilon}_{n+1}^p \right)

      \bm{\varepsilon}_{n+1}^p = 
         \begin{cases}
            \bm{\varepsilon}^{p}_{tr} & f\left(\bm{\sigma}_{tr},\bm{\alpha}_{tr}\right)\le0\\
            \bm{\varepsilon}^{p}_{tr}+\mathbf{g}\left( \bm{\sigma}_{n+1}, \bm{\alpha}_{n+1}, T_{n+1} \right)\Delta\gamma_{n+1} & f\left(\bm{\sigma}_{tr},\bm{\alpha}_{tr}\right)>0
         \end{cases}

      \bm{\alpha}_{n+1} = 
         \begin{cases}
            \bm{\alpha}_{tr} & f\left(\bm{\sigma}_{tr},\bm{\alpha}_{tr}\right)\le0\\
            \bm{\alpha}_{tr}+\mathbf{h}\left( \bm{\sigma}_{n+1}, \bm{\alpha}_{n+1}, T_{n+1} \right)\Delta\gamma_{n+1} & f\left(\bm{\sigma}_{tr},\bm{\alpha}_{tr}\right)>0
         \end{cases}

   Solving for :math:`\Delta \gamma_{n+1}` such that

   .. math::
      f\left(\bm{\sigma}_{n+1}, \bm{\alpha}_{n+1} \right) = 0

In these equations :math:`f` is a yield function, :math:`\mathbf{g}` is
a flow function, evaluated at the next state, and :math:`\mathbf{h}` is 
the rate of evolution for the history variables, evaluated at the next
state.
NEML integrates all three of these functions into a :doc:`../ri_flow`
interface.

If the step is plastic the stress update is solved through fully-implicit 
backward Euler integration.
The algorithmic tangent is then computed using an implicit function scheme.
The work and energy are integrated with a trapezoid rule from the final values
of stress and plastic strain.

This model maintains a vector of history variables defined by the
model's :doc:`../ri_flow` interface.

At the end of the step the model (optionally) checks to ensure the step
met the Kuhn-Tucker conditions

.. math::

   \Delta \gamma_{n+1} \ge 0

   f\left(\bm{\sigma}_{n+1}, \bm{\alpha}_{n+1} \right) \le 0

   \Delta \gamma_{n+1} f\left(\bm{\sigma}_{n+1}, \bm{\alpha}_{n+1} \right) = 0. 

Parameters
----------

.. csv-table::
   :header: "Parameter", "Object type", "Description", "Default"
   :widths: 12, 30, 50, 8

   ``elastic``   , :cpp:class:`neml::LinearElasticModel`     , Temperature dependent elastic constants, No
   ``surface``   , :cpp:class:`neml::RateIndependentFlowRule`, Flow rule interface                    , No
   ``alpha``     , :cpp:class:`neml::Interpolate`            , Temperature dependent instantaneous CTE, ``0.0``
   ``tol``       , :code:`double`                 , Integration tolerance                  , ``1.0e-8``
   ``miter``     , :code:`int`                    , Maximum number of integration iters    , ``50``
   ``verbose``   , :code:`bool`                   , Print lots of convergence info         , ``false``
   ``kttol``     , :code:`double`                 , Tolerance on the Kuhn-Tucker conditions, ``1.0e-2``
   ``check_kt``  , :code:`bool`                   , Flag to actually check KT              , ``false``

Class description
-----------------

.. doxygenclass:: neml::SmallStrainRateIndependentPlasticity
   :members:
   :undoc-members:
