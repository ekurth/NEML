#ifndef OBJECTS_H
#define OBJECTS_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <stdexcept>

#include "boost/variant.hpp"

namespace neml {

class NEMLObject {

};

// This version supports the following types of objects as parameters:
//    double
//    int
//    bool
//    vector<double>
//    NEMLObject

// This black magic lets us store parameters in a unified map
typedef boost::variant<double, int, bool, std::vector<double>, 
        std::shared_ptr<NEMLObject>> param_type;
// This is the enum name we assign to each type for the "external" interfaces
// to use in reconstructing a type from data
enum ParamType {
  TYPE_DOUBLE       = 0,
  TYPE_INT          = 1,
  TYPE_BOOL         = 2,
  TYPE_VEC_DOUBLE   = 3,
  TYPE_NEML_OBJECT  = 4
};
// This black magic lets us map the actual type of each parameter to the enum
template <class T> constexpr ParamType GetParamType();
template <> constexpr ParamType GetParamType<double>() {return TYPE_DOUBLE;}
template <> constexpr ParamType GetParamType<int>() {return TYPE_INT;}
template <> constexpr ParamType GetParamType<bool>() {return TYPE_BOOL;}
template <> constexpr ParamType GetParamType<std::vector<double>>() 
{return TYPE_VEC_DOUBLE;}
template <> constexpr ParamType GetParamType<std::shared_ptr<NEMLObject>>()
{return TYPE_NEML_OBJECT;}
template <> constexpr ParamType GetParamType<NEMLObject>()
{return TYPE_NEML_OBJECT;}

/// Parameters for objects created through the NEMLObject interface
class ParameterSet {
 public:
  /// Default constructor, needed to push onto stack
  ParameterSet();
  /// Constructor giving object type
  ParameterSet(std::string type);
  
  virtual ~ParameterSet();
  
  /// Return the type of object you're supposed to create
  const std::string & type() const;
  
  /// Add a generic parameter with no default
  template<typename T>
  void add_parameter(std::string name)
  {
    param_names_.push_back(name);
    param_types_[name] = GetParamType<T>();
  }
  
  /// Immediately assign an input of the right type to a parameter
  void assign_parameter(std::string name, param_type value)
  {
    params_[name] = value;
  }
  
  /// Add a generic parameter with a default
  template<typename T>
  void add_optional_parameter(std::string name, param_type value)
  {
    add_parameter<T>(name);
    assign_parameter(name, value);
  }
  
  /// Get a parameter of the given name and type
  template<typename T>
  T get_parameter(std::string name)
  {
    resolve_objects_();
    return boost::get<T>(params_[name]);
  }
  
  /// Assign a parameter set to be used to create an object later
  void assign_defered_parameter(std::string name, ParameterSet value);
  
  /// Helper method to get a NEMLObject and cast it to subtype in one go 
  template<typename T>
  std::shared_ptr<T> get_object_parameter(std::string name) 
  {
    return std::dynamic_pointer_cast<T>(get_parameter<std::shared_ptr<NEMLObject>>(name));
  };
  
  /// Get the type of parameter
  ParamType get_object_type(std::string name);

  /// Check to make sure this parameter set is ready to go
  bool fully_assigned();

 private:
  /// Run down the chain of deferred objects and actually construct them
  void resolve_objects_();
  
  std::string type_;
  
  std::vector<std::string> param_names_;
  std::map<std::string, ParamType> param_types_;
  std::map<std::string, param_type> params_;
  std::map<std::string, ParameterSet> defered_params_;
};

/// Factory that produces NEMLObjects from ParameterSets
class Factory {
 public:
  /// Provide a valid parameter set for the object type
  ParameterSet provide_parameters(std::string type);
  
  /// Create an object from the parameter set
  std::shared_ptr<NEMLObject> create(ParameterSet & params);

  /// Register a type with an identifier, create method, and parameter set
  void register_type(std::string type, 
                     std::function<std::shared_ptr<NEMLObject>(ParameterSet &)> creator,
                     std::function<ParameterSet()> setup);

  /// Static factor instance
  static Factory * Creator();

 private:
  std::map<std::string, std::function<std::shared_ptr<NEMLObject>(ParameterSet &)>> creators_;
  std::map<std::string, std::function<ParameterSet()>> setups_;
};

/// Little object used for auto registration
template<typename T>
class Register {
 public:
  Register()
  {
    Factory::Creator()->register_type(T::type(), &T::initialize, &T::parameters);
  }
};

} //namespace neml

#endif // OBJECTS_H
