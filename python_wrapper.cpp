#include "chemo_example.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(chemo) {
  numpy::initialize();
  class_<Simulation>("Simulation", init<size_t, int>())
      .def("integrate", &Simulation::integrate)
      .def("get_positions", &Simulation::get_positions)
      .def("get_starting", &Simulation::get_starting)
      .def("get_drift", &Simulation::get_drift)
      .def("get_conc", &Simulation::get_conc)
      .def("get_type", &Simulation::get_type)
      .def("get_grid_conc", &Simulation::get_grid_conc)
      .def("get_grid_rhoalpha", &Simulation::get_grid_rhoalpha)
      .def("get_grid_rhobeta", &Simulation::get_grid_rhobeta);
}
