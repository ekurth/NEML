Power law damage
================

Overview
--------

This object implements a "standard" damage model proportional to a power law in stress and directly
to the effective inelastic strain.
The damage function is

.. math::

   w = A \sigma_{eff}^n

   \sigma_{eff} = \sqrt{\frac{3}{2} \operatorname{dev}\left(\bm{\sigma}\right):\operatorname{dev}\left(\bm{\sigma}\right)}.

The standard damage model multiplies this function by the inelastic
strain rate in computing the damage update.

Parameters
----------

.. csv-table::
   :header: "Parameter", "Object type", "Description", "Default"
   :widths: 12, 30, 50, 8

   ``elastic``, :cpp:class:`neml::LinearElasticModel`, Elasticity model, No
   ``A``, :cpp:class:`neml::Interpolate`, Prefactor, No
   ``a``, :cpp:class:`neml::Interpolate`, Stress exponent, No
   ``base``, :cpp:class:`neml::NEMLModel_sd`, Base material model, No
   ``alpha``, :cpp:class:`neml::Interpolate`, Thermal expansion coefficient, ``0.0``
   ``tol``, :code:`double`, Solver tolerance, ``1.0e-8``
   ``miter``, :code:`int`, Maximum solver iterations, ``50``
   ``verbose``, :code:`bool`, Verbosity flag, ``false``

Class description
-----------------

.. doxygenclass:: neml::NEMLPowerLawDamagedModel_sd
   :members:
   :undoc-members:
