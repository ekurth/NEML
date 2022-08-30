#pragma once

#include "solvers.h"
#include "objects.h"
#include "elasticity.h"
#include "ri_flow.h"
#include "visco_flow.h"
#include "general_flow.h"
#include "interpolate.h"
#include "creep.h"
#include "history.h"

#include "windows.h"

#include <cstddef>
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>

namespace neml {

/// NEML material model interface definitions
//  All material models inherit from this base class.  It defines interfaces
//  and provides the methods for reading in material parameters.
class NEML_EXPORT NEMLModel: public HistoryNEMLObject {
  public:
   NEMLModel(ParameterSet & params);
   virtual ~NEMLModel() {};

   /// Store model to an XML file
   virtual void save(std::string file_name, std::string model_name);
  
   /// Setup the history
   virtual void populate_hist(History & history) const;
   /// Initialize the history
   virtual void init_hist(History & history) const;

   /// Setup the actual evolving state
   virtual void populate_state(History & history) const = 0;
   /// Initialize the actual evolving state
   virtual void init_state(History & history) const = 0;

   /// Setup any static state
   virtual void populate_static(History & history) const;
   /// Initialize any static state
   virtual void init_static(History & history) const;

   /// Raw data small strain update interface
   virtual void update_sd(
       const double * const e_np1, const double * const e_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       double * const s_np1, const double * const s_n,
       double * const h_np1, const double * const h_n,
       double * const A_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);

   /// Small strain update, wrapped objects
   virtual void update_sd_interface(
       const Symmetric & e_np1, const Symmetric & e_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & s_np1, Symmetric & s_n,
       History & h_np1, const History & h_n,
       SymSymR4 & A_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n) {}; // Fix after all updated

   /// Raw data large strain incremental update
   virtual void update_ld_inc(
       const double * const d_np1, const double * const d_n,
       const double * const w_np1, const double * const w_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       double * const s_np1, const double * const s_n,
       double * const h_np1, const double * const h_n,
       double * const A_np1, double * const B_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);

   /// Large strain incremental update, wrapped objects
   virtual void update_ld_inc_interface(
       const Symmetric & d_np1, const Symmetric & d_n,
       const Skew & w_np1, const Skew & w_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & s_np1, const Symmetric & s_n,
       History & h_np1, const History & h_n,
       SymSymR4 & A_np1, SymSkewR4 & B_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n) {}; // Fix after all updated
  
   /// Instantaneous thermal expansion coefficient as a function of temperature
   virtual double alpha(double T) const = 0;
   /// Elastic strain for a given stress, temperature, and history state
   virtual void elastic_strains(const double * const s_np1,
                               double T_np1, const double * const h_np1,
                               double * const e_np1) const;
   /// Nice interface for the elastic strain calculation
   virtual Symmetric elastic_strains_interface(const Symmetric & s_np1, 
                                               double T_np1, const History & h_np1) const;

   /// Used to find the damage value from the history
   virtual double get_damage(const double *const h_np1);
   /// Used to determine if element should be deleted
   virtual bool should_del_element(const double *const h_np1);
   /// Used to determine if this is a damage model
   virtual bool is_damage_model() const;

   /// Number of actual internal variables
   size_t nstate() const;
   /// Number of static variables
   size_t nstatic() const;

   /// Quickly setup state
   History gather_state_(double * data) const;
   History gather_state_(const double * data) const;
   History gather_blank_state_() const;

  protected:
   virtual void cache_history_();
   /// Split internal variables into static and actual parts
   std::tuple<History,History> split_state(const History & h) const;

  protected:
   History stored_state_;
   History stored_static_;
};

/// Large deformation incremental update model
class NEML_EXPORT NEMLModel_ldi: public NEMLModel {
  public:
    NEMLModel_ldi(ParameterSet & params);

   /// Small strain update, wrapped objects
   virtual void update_sd_interface(
       const Symmetric & e_np1, const Symmetric & e_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & s_np1, Symmetric & s_n,
       History & h_np1, const History & h_n,
       SymSymR4 & A_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);
};

/// Small deformation stress update
class NEML_EXPORT NEMLModel_sd: public NEMLModel {
  public:
    /// All small strain models use small strain elasticity and CTE
    NEMLModel_sd(ParameterSet & params);

   /// Small strain update, wrapped objects
   virtual void update_sd_interface(
       const Symmetric & e_np1, const Symmetric & e_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & s_np1, Symmetric & s_n,
       History & h_np1, const History & h_n,
       SymSymR4 & A_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n); 

   /// The small strain stress update interface with just the state variables
   virtual void update_sd_state(
       const Symmetric & E_np1, const Symmetric & E_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & S_np1, const Symmetric & S_n,
       History & H_np1, const History & H_n,
       SymSymR4 & AA_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n) = 0;

   /// Large strain incremental update
   virtual void update_ld_inc_interface(
       const Symmetric & d_np1, const Symmetric & d_n,
       const Skew & w_np1, const Skew & w_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & s_np1, const Symmetric & s_n,
       History & h_np1, const History & h_n,
       SymSymR4 & A_np1, SymSkewR4 & B_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);

   /// Setup any static state
   virtual void populate_static(History & history) const;
   /// Initialize any static state
   virtual void init_static(History & history) const;

   /// Provide the instantaneous CTE
   virtual double alpha(double T) const;
   /// Returns the elasticity model, for sub-objects that want to use it
   const std::shared_ptr<const LinearElasticModel> elastic() const;

   /// Return the elastic strains
   virtual Symmetric elastic_strains_interface(const Symmetric & s_np1, 
                                               double T_np1, const History & h_np1) const;

   /// Used to override the linear elastic model to match another object's
   virtual void set_elastic_model(std::shared_ptr<LinearElasticModel> emodel);

  private:
   void calc_tangent_(const Symmetric & D, const Skew & W,
                     const SymSymR4 & C, const Symmetric & S,
                     SymSymR4 & A, SymSkewR4 & B);

  protected:
   std::shared_ptr<LinearElasticModel> elastic_;

  private:
   std::shared_ptr<Interpolate> alpha_;
   bool truesdell_;
};

/// Adaptive integration, tangent using the usual trick
class SubstepModel_sd: public NEMLModel_sd, public Solvable {
 public:
  SubstepModel_sd(ParameterSet & params);

  /// Complete substep update
  virtual void update_sd_state(
       const Symmetric & E_np1, const Symmetric & E_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & S_np1, const Symmetric & S_n,
       History & H_np1, const History & H_n,
       SymSymR4 & AA_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);

  /// Single step update
  virtual void update_step(
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n, double t_np1, double t_n,
      Symmetric & s_np1, const Symmetric & s_n,
      History & h_np1, const History & h_n,
      double * const A, double * const E,
      double & u_np1, double u_n,
      double & p_np1, double p_n);

  /// Setup the trial state
  virtual std::unique_ptr<TrialState> setup(
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n) = 0;

  /// Ignore update and take an elastic step
  virtual bool elastic_step(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n) = 0;

  /// Interpret the x vector
  virtual void update_internal(
      const double * const x,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n, 
      double t_np1, double t_n,
      Symmetric & s_np1, const Symmetric & s_n,
      History & h_np1, const History & h_n) = 0;

  /// Minus the partial derivative of the residual with respect to the strain
  virtual void strain_partial(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_np1, const Symmetric & s_n,
      const History & h_np1, const History & h_n,
      double * de) = 0;

  /// Do the work calculation
  virtual void work_and_energy(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_np1, const Symmetric & s_n,
      const History & h_np1, const History & h_n,
      double & u_np1, double u_n,
      double & p_np1, double p_n);

 protected:
  double rtol_, atol_;
  int miter_;
  bool verbose_, linesearch_;
  int max_divide_;
  bool force_divide_;
};

/// Small strain linear elasticity
//  This is generally only used as a basic test
class NEML_EXPORT SmallStrainElasticity: public NEMLModel_sd {
 public:
  /// Parameters are the minimum: an elastic model and a thermal expansion
  SmallStrainElasticity(ParameterSet & params);

  /// Type for the object system
  static std::string type();
  /// Setup parameters for the object system
  static ParameterSet parameters();
  /// Initialize from a parameter set
  static std::unique_ptr<NEMLObject> initialize(ParameterSet & params);

  /// Small strain stress update
  virtual void update_sd_state(
       const Symmetric & E_np1, const Symmetric & E_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & S_np1, const Symmetric & S_n,
       History & H_np1, const History & H_n,
       SymSymR4 & AA_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);
  
  /// Populate internal variables (none)
  virtual void populate_state(History & h) const;

  /// Initialize history (none to setup)
  virtual void init_state(History & h) const;
};

static Register<SmallStrainElasticity> regSmallStrainElasticity;

/// Small strain perfect plasticity trial state
//  Store data the solver needs and can be passed into solution interface
class SSPPTrialState : public TrialState {
 public:
  SSPPTrialState(const Symmetric & e_np1, const Symmetric & ep_n,
                 const Symmetric & s_tr, const SymSymR4 & C,
                 double ys, double T) :
      e_np1(e_np1), ep_n(ep_n), s_tr(s_tr), C(C), ys(ys), T(T)
  {};
  Symmetric e_np1;    // Next strain
  Symmetric ep_n;     // Previous plastic strain
  Symmetric s_tr;     // Trial stress
  SymSymR4 C;         // Elastic stiffness
  double ys, T;       // Yield strength and temperature
};

/// Small strain rate independent plasticity trial state
class SSRIPTrialState : public TrialState {
 public:
  SSRIPTrialState(const Symmetric & e_np1, const Symmetric & ep_tr,
                  const Symmetric & s_tr, const SymSymR4 & C,
                  const History & h_tr, double T) :
      e_np1(e_np1), ep_tr(ep_tr), s_tr(s_tr), C(C), 
      h_tr(h_tr), T(T)
  {};

  Symmetric e_np1;        // Next strain
  Symmetric ep_tr;        // Trial plastic strain
  Symmetric s_tr;         // Trial stress
  SymSymR4 C;             // Elastic stiffness
  History h_tr;           // Trial history
  double T;               // Temperature
};

/// Small strain creep+plasticity trial state
class SSCPTrialState : public TrialState {
 public:
  SSCPTrialState(const Symmetric & ep_strain, const Symmetric & e_n,
                 const Symmetric & e_np1, const Symmetric & s_n,
                 double T_n, double T_np1, double t_n, double t_np1,
                 const History & h_n) :
      ep_strain(ep_strain), e_n(e_n), e_np1(e_np1), s_n(s_n), 
      T_n(T_n), T_np1(T_np1), t_n(t_n), t_np1(t_np1), h_n(h_n)
  {};
  Symmetric ep_strain;            // Current plastic strain
  Symmetric e_n, e_np1;           // Previous and next total strain
  Symmetric s_n;                  // Previous stress
  double T_n, T_np1, t_n, t_np1;  // Next and previous time and temperature
  History h_n;                    // Previous history vector
};

/// General inelastic integrator trial state
class GITrialState : public TrialState {
 public:
  GITrialState(const Symmetric & e_dot, const Symmetric & s_n, 
               const Symmetric & s_guess, const History & h_n,
               double T, double Tdot, double dt) :
      e_dot(e_dot), s_n(s_n), s_guess(s_guess), h_n(h_n),
      T(T), Tdot(Tdot), dt(dt)
  {};
  Symmetric e_dot;      // Strain rate
  Symmetric s_n;        // Previous stress
  Symmetric s_guess;    // Guess at next stress
  History h_n;          // Previous history
  double T, Tdot, dt;   // Temperature, temperature rate, time increment
};

/// Small strain, associative, perfect plasticity
//    Algorithm is generalized closest point projection.
//    This degenerates to radial return for models where the gradient of
//    the yield surface is constant along lines from the origin to a point
//    in stress space outside the surface (i.e. J2).

class NEML_EXPORT SmallStrainPerfectPlasticity: public SubstepModel_sd {
 public:
  /// Parameters: elastic model, yield surface, yield stress, CTE,
  /// integration tolerance, maximum number of iterations,
  /// verbosity flag, and the maximum number of adaptive subdivisions
  SmallStrainPerfectPlasticity(ParameterSet & params);

  /// Type for the object system
  static std::string type();
  /// Parameters for the object system
  static ParameterSet parameters();
  /// Setup from a ParameterSet
  static std::unique_ptr<NEMLObject> initialize(ParameterSet & params);
  
  /// Populate the internal variables (nothing)
  virtual void populate_state(History & h) const;
  /// Initialize history (nothing to do)
  virtual void init_state(History & h) const;
  
  /// Number of nonlinear equations to solve in the integration
  virtual size_t nparams() const;
  /// Setup an initial guess for the nonlinear solution
  virtual void init_x(double * const x, TrialState * ts);
  /// Integration residual and jacobian equations
  virtual void RJ(const double * const x, TrialState * ts, double * const R,
                 double * const J);

  /// Setup the trial state
  virtual std::unique_ptr<TrialState> setup(
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n);
  
  /// Take an elastic step
  virtual bool elastic_step(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n);

  /// Interpret the x vector
  virtual void update_internal(
      const double * const x,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n, 
      double t_np1, double t_n,
      Symmetric & s_np1, const Symmetric & s_n,
      History & h_np1, const History & h_n);

  /// Minus the partial derivative of the residual with respect to the strain
  virtual void strain_partial(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_np1, const Symmetric & s_n,
      const History & h_np1, const History & h_n,
      double * de);

  /// Helper to return the yield stress
  double ys(double T) const;

 private:
  std::shared_ptr<YieldSurface> surface_;
  std::shared_ptr<Interpolate> ys_;
};

static Register<SmallStrainPerfectPlasticity> regSmallStrainPerfectPlasticity;

/// Small strain, rate-independent plasticity
//    The algorithm used here is generalized closest point projection
//    for associative flow models.  For non-associative models the algorithm
//    may theoretically fail the discrete Kuhn-Tucker conditions, even
//    putting aside convergence issues on the nonlinear solver.
class NEML_EXPORT SmallStrainRateIndependentPlasticity: public SubstepModel_sd {
 public:
  /// Parameters: elasticity model, flow rule, CTE, solver tolerance, maximum
  /// solver iterations, verbosity flag, tolerance on the Kuhn-Tucker conditions
  /// check, and a flag on whether the KT conditions should be evaluated
  SmallStrainRateIndependentPlasticity(ParameterSet & params);

  /// Type for the object system
  static std::string type();
  /// Parameters for the object system
  static ParameterSet parameters();
  /// Setup from a ParameterSet
  static std::unique_ptr<NEMLObject> initialize(ParameterSet & params);
  
  /// Populate internal variables
  virtual void populate_state(History & h) const;
  /// Initialize history at time zero
  virtual void init_state(History & h) const;

  /// Setup the trial state
  virtual std::unique_ptr<TrialState> setup(
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n);

  /// Ignore update and take an elastic step
  virtual bool elastic_step(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n);

  /// Interpret the x vector
  virtual void update_internal(
      const double * const x,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n, 
      double t_np1, double t_n,
      Symmetric & s_np1, const Symmetric & s_n,
      History & h_np1, const History & h_n);

  /// Minus the partial derivative of the residual with respect to the strain
  virtual void strain_partial(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_np1, const Symmetric & s_n,
      const History & h_np1, const History & h_n,
      double * de);

  /// Number of solver parameters
  virtual size_t nparams() const;
  /// Setup an iteration vector in the solver
  virtual void init_x(double * const x, TrialState * ts);
  /// Solver function returning the residual and jacobian of the nonlinear
  /// system of equations integrating the model
  virtual void RJ(const double * const x, TrialState * ts, double * const R,
                 double * const J);

  /// Return the elastic model for subobjects
  const std::shared_ptr<const LinearElasticModel> elastic() const;

 private:
  std::shared_ptr<RateIndependentFlowRule> flow_;
};

static Register<SmallStrainRateIndependentPlasticity> regSmallStrainRateIndependentPlasticity;

/// Small strain, rate-independent plasticity + creep
//  Uses a combined iteration of a rate independent plastic + creep model
//  to solver overall update
class NEML_EXPORT SmallStrainCreepPlasticity: public NEMLModel_sd, public Solvable {
 public:
  /// Parameters are an elastic model, a base NEMLModel_sd, a CreepModel,
  /// the CTE, a solution tolerance, the maximum number of nonlinear
  /// iterations, a verbosity flag, and a scale factor to regularize
  /// the nonlinear equations.
  SmallStrainCreepPlasticity(ParameterSet & params);

  /// Type for the object system
  static std::string type();
  /// Setup parameters for the object system
  static ParameterSet parameters();
  /// Initialize from a parameter set
  static std::unique_ptr<NEMLObject> initialize(ParameterSet & params);

  /// Small strain stress update
  virtual void update_sd_state(
       const Symmetric & E_np1, const Symmetric & E_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & S_np1, const Symmetric & S_n,
       History & H_np1, const History & H_n,
       SymSymR4 & AA_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);
  
  /// Populate list of internal variables
  virtual void populate_state(History & hist) const;
  /// Passes call for initial history to base model
  virtual void init_state(History & hist) const;

  /// The number of parameters in the nonlinear equation
  virtual size_t nparams() const;
  /// Initialize the nonlinear solver
  virtual void init_x(double * const x, TrialState * ts);
  /// Residual equation to solve and corresponding jacobian
  virtual void RJ(const double * const x, TrialState * ts, double * const R,
                 double * const J);

  /// Setup a trial state from known information
  std::unique_ptr<SSCPTrialState> make_trial_state(
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n, double t_np1, double t_n,
      const Symmetric & s_n, const History & h_n);

  /// Set a new elastic model
  virtual void set_elastic_model(std::shared_ptr<LinearElasticModel> emodel);

 private:
  SymSymR4 form_tangent_(const SymSymR4 & A, const SymSymR4 & B);

 private:
  std::shared_ptr<NEMLModel_sd> plastic_;
  std::shared_ptr<CreepModel> creep_;

  double rtol_, atol_, sf_;
  int miter_;
  bool verbose_, linesearch_;
};

static Register<SmallStrainCreepPlasticity> regSmallStrainCreepPlasticity;

/// Small strain general integrator
//    General NR one some stress rate + history evolution rate
//
class NEML_EXPORT GeneralIntegrator: public SubstepModel_sd {
 public:
  /// Parameters are an elastic model, a general flow rule,
  /// the CTE, the integration tolerance, the maximum
  /// nonlinear iterations, a verbosity flag, and the
  /// maximum number of subdivisions for adaptive integration
  GeneralIntegrator(ParameterSet & params);

  /// Type for the object system
  static std::string type();
  /// Parameters for the object system
  static ParameterSet parameters();
  /// Setup from a ParameterSet
  static std::unique_ptr<NEMLObject> initialize(ParameterSet & params);

  /// Setup the trial state
  virtual std::unique_ptr<TrialState> setup(
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n);
  
  /// Take an elastic step
  virtual bool elastic_step(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_n,
      const History & h_n);

  /// Interpret the x vector
  virtual void update_internal(
      const double * const x,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n, 
      double t_np1, double t_n,
      Symmetric & s_np1, const Symmetric & s_n,
      History & h_np1, const History & h_n);

  /// Minus the partial derivative of the residual with respect to the strain
  virtual void strain_partial(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_np1, const Symmetric & s_n,
      const History & h_np1, const History & h_n,
      double * de);
  
  /// Need special call for dissipation
  virtual void work_and_energy(
      const TrialState * ts,
      const Symmetric & e_np1, const Symmetric & e_n,
      double T_np1, double T_n,
      double t_np1, double t_n,
      const Symmetric & s_np1, const Symmetric & s_n,
      const History & h_np1, const History & h_n,
      double & u_np1, double u_n,
      double & p_np1, double p_n);

  /// Populate internal variables
  virtual void populate_state(History & hist) const;
  /// Initialize the history at time zero
  virtual void init_state(History & hist) const;

  /// Number of nonlinear equations
  virtual size_t nparams() const;
  /// Initialize a guess for the nonlinear iterations
  virtual void init_x(double * const x, TrialState * ts);
  /// The residual and jacobian for the nonlinear solve
  virtual void RJ(const double * const x, TrialState * ts,
                 double * const R, double * const J);

  /// Set a new elastic model
  virtual void set_elastic_model(std::shared_ptr<LinearElasticModel> emodel);

 private:
  std::shared_ptr<GeneralFlowRule> rule_;
  bool skip_first_;
};

static Register<GeneralIntegrator> regGeneralIntegrator;

/// Combines multiple small strain integrators based on regimes of
/// rate-dependent behavior.
//
//  This model uses the idea from Kocks & Mecking of a normalized activation
//  energy to call different integrators depending on the combination of
//  temperature and strain rate.
//
//  A typical use case would be switching from rate-independent to rate
//  dependent behavior based on a critical activation energy cutoff point
//
//  A user provides a vector of models (length n) and a corresponding vector
//  of normalized activation energies (length n-1) dividing the response into
//  segments.  All the models must have compatible hardening -- the history
//  is just going to be blindly passed between the models.
//
class NEML_EXPORT KMRegimeModel: public NEMLModel_sd {
 public:
  /// Parameters are an elastic model, a vector of valid NEMLModel_sd objects,
  /// the transition activation energies, the Boltzmann constant in appropriate
  /// units, a Burgers vector for normalization, a reference strain rate,
  /// and the CTE.
  KMRegimeModel(ParameterSet & params);

  /// Type for the object system
  static std::string type();
  /// Parameters for the object system
  static ParameterSet parameters();
  /// Setup from a ParameterSet
  static std::unique_ptr<NEMLObject> initialize(ParameterSet & params);

  /// The small strain stress update
  virtual void update_sd_state(
       const Symmetric & E_np1, const Symmetric & E_n,
       double T_np1, double T_n,
       double t_np1, double t_n,
       Symmetric & S_np1, const Symmetric & S_n,
       History & H_np1, const History & H_n,
       SymSymR4 & AA_np1,
       double & u_np1, double u_n,
       double & p_np1, double p_n);

  /// Populate internal variables
  virtual void populate_state(History & hist) const;
  /// Initialize history at time zero
  virtual void init_state(History & hist) const;

  /// Set a new elastic model
  virtual void set_elastic_model(std::shared_ptr<LinearElasticModel> emodel);

 private:
  double activation_energy_(const Symmetric & e_np1, const Symmetric & e_n,
                            double T_np1, double t_np1, double t_n);

 private:
  std::vector<std::shared_ptr<NEMLModel_sd>> models_;
  std::vector<double> gs_;
  double kboltz_, b_, eps0_;
};

static Register<KMRegimeModel> regKMRegimeModel;

/// Useful helper to calculate work and energy with the trapezoid rule
std::tuple<double,double> trapezoid_energy(
    const Symmetric & e_np1, const Symmetric & e_n,
    const Symmetric & ep_np1, const Symmetric & ep_n,
    const Symmetric & s_np1, const Symmetric & s_n);

} // namespace neml
