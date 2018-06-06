#include <Python.h>
#include <Eigen/Sparse>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
namespace p = boost::python;
namespace np = boost::python::numpy;

#include "Aboria.h"
using namespace Aboria;

class Simulation {
    const double PI = boost::math::constants::pi<double>();

    // [aboria type definitions
    ABORIA_VARIABLE(conc, double, "conc");
    ABORIA_VARIABLE(drift, vdouble2, "drift");
    ABORIA_VARIABLE(starting, vdouble2, "starting_position");
    ABORIA_VARIABLE(next_position, vdouble2, "next position");
    ABORIA_VARIABLE(type, uint8_t, "type");
    typedef Particles<std::tuple<type, drift, conc, starting, next_position>, 2>
        Particles_t;
    typedef typename Particles_t::position position;
    // ]

    typedef Eigen::SparseMatrix<double> SparseMatrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 2> Vector2;

    // data containers
    Particles_t particles;
    Vector c;
    Vector rhoalpha;
    Vector rhobeta;
    Vector2 grad_c;
    SparseMatrix D2;
    SparseMatrix D1x;
    SparseMatrix D1y;

    // parameters

    // domain and grid
    int Nc;  // number of grid points
    int Nc2;
    double L;       // physical domain size
    vdouble2 min;   // minimum domain extent
    vdouble2 max;   // maximum domain extent
    double hc;      // lattice spacing for chemical
    bool periodic;  // periodicity of domain

    // diffusions, chemotaxis constants
    double Db;    // type beta diffusion coefficient
    double Da;    // type alpha diffusion coefficient
    double Dsum;  // sum of diffusion coefficients
    double Dc;    // chemical diffusion coefficient
    double chi;   // chemotaxis affinity

    // reaction rates
    double ka;   // chemo production
    double kb;   // chemo consumption
    double gam;  // chemo degradation
    double rb;   // rate beta to alpha type
    double ra;   // rate alpha to beta type

    unsigned int Nb;  // initial number of beta particles
    unsigned int Na;  // initial number of alpha particles
    unsigned int N;   // total number of particles (constant)

    double c0;  // initial chemical concentration

    // cell-cell interaction constants
    double cutoff;   // interaction cutoff
    double epsilon;  // interaction range
    double inter;    // interaction strength

    double timestep_ratio;
    double mean_s;
    double dt;

   public:
    Simulation(const size_t sample, const int simulation_type = 0) {
        // setup parameter values
        Nc = 52;
        Nc2 = Nc * Nc;
        L = 1.0;
        min = vdouble2::Constant(-L / 2);
        max = vdouble2::Constant(L / 2);
        hc = L / (Nc - 1);

        epsilon = 0.02;
        inter = 1.0;

        std::cout << "simulation_type = " << simulation_type << std::endl;
        switch (simulation_type) {
            case 0:
                Na = 100;
                Nb = 100;
                ra = 10.0;
                rb = 0.0;
                periodic = false;
                Db = 1.0;
                Da = 0.1;
                Dc = 1.0;
                chi = 1.0;
                ka = 0.1;
                kb = 0.03;
                gam = 0.5;
                c0 = 0.0;
                break;
            case 1:
                Na = 100;
                Nb = 0;
                ra = 0.0;
                rb = 0.0;
                periodic = true;
                Db = 1.0;
                Da = 1.0;
                Dc = 0.0;
                chi = 10.0;
                ka = 0.0;
                kb = 0.0;
                gam = 0.0;
                c0 = 1.0;
                break;
            case 2:
                Na = 0;
                Nb = 100;
                ra = 0.0;
                rb = 0.0;
                periodic = true;
                Db = 1.0;
                Da = 1.0;
                Dc = 0.0;
                chi = 10.0;
                ka = 0.0;
                kb = 0.0;
                gam = 0.0;
                c0 = 1.0;
                break;
            case 3:
                Na = 50;
                Nb = 50;
                ra = 10.0;
                rb = 0.0;
                periodic = true;
                Db = 1.0;
                Da = 1.0;
                Dc = 0.0;
                chi = 10.0;
                ka = 0.0;
                kb = 0.0;
                gam = 0.0;
                rb = 0.0;
                c0 = 1.0;
                break;
        }
        N = Na + Nb;
        Dsum = Da + Db;
        cutoff = 8 * epsilon;
        timestep_ratio = 0.23;
        mean_s = timestep_ratio * std::max(epsilon, 0.01);
        dt = std::pow(mean_s, 2) / (4 * std::max(Da, Db));

        // initialise simulation
        initialise(sample);
    }
    void initialise(const size_t sample);
    void integrate(const double time);

    // [export eigen vector to python
    template <typename V>
    p::object get_vector(V &v, const size_t N) {
        np::dtype dt = np::dtype::get_builtin<double>();
        p::tuple shape = p::make_tuple(v.size(), N);
        p::tuple stride = p::make_tuple(sizeof(double) * N, sizeof(double));
        p::object own;
        return np::from_data(reinterpret_cast<double *>(v.data()), dt, shape,
                             stride, own);
    }
    // ]

    p::object get_grid_conc() { return get_vector(c, 1); }
    p::object get_grid_grad_c() { return get_vector(grad_c, 2); }
    p::object get_grid_rhoalpha() { return get_vector(rhoalpha, 1); }

    p::object get_grid_rhobeta() { return get_vector(rhobeta, 1); }

    // [export aboria vector variable to python
    template <typename V>
    p::object get_particle_vector() {
        using data_t = typename V::value_type::value_type;
        const size_t N = V::value_type::size;
        np::dtype dt = np::dtype::get_builtin<data_t>();
        p::tuple shape = p::make_tuple(particles.size(), N);
        p::tuple stride = p::make_tuple(sizeof(data_t) * N, sizeof(data_t));
        p::object own;
        return np::from_data(
            reinterpret_cast<double *>(get<V>(particles).data()), dt, shape,
            stride, own);
    }
    // ]

    // [export aboria scalar variable to python
    template <typename V>
    p::object get_particle_scalar() {
        using data_t = typename V::value_type;
        np::dtype dt = np::dtype::get_builtin<data_t>();
        p::tuple shape = p::make_tuple(particles.size());
        p::tuple stride = p::make_tuple(sizeof(data_t));
        p::object own;
        return np::from_data(
            reinterpret_cast<double *>(get<V>(particles).data()), dt, shape,
            stride, own);
    }
    // ]

    p::object get_positions() { return get_particle_vector<position>(); }
    p::object get_starting() { return get_particle_vector<starting>(); }
    p::object get_drift() { return get_particle_vector<drift>(); }
    p::object get_conc() { return get_particle_scalar<conc>(); }
    p::object get_type() { return get_particle_scalar<type>(); }

   private:
    void insertCoefficient(int id, int i, int j, double w, int n,
                           std::vector<Eigen::Triplet<double>> &coeffs) {
        if (i == -1)
            i = 1;
        else if (i == Nc)
            i = Nc - 2;

        if (j == -1)
            j = 1;
        else if (j == Nc)
            j = Nc - 2;

        const int id1 = i * Nc + j;
        coeffs.push_back(Eigen::Triplet<double>(id, id1, w));
    }
};
