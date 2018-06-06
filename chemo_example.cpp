#include "chemo_example.h"
#include <stdexcept>

void Simulation::initialise(const size_t sample) {
    // allocate chemical grid and matrices
    c = Vector::Zero(Nc2);
    rhoalpha = Vector::Zero(Nc2);
    rhobeta = Vector::Zero(Nc2);
    grad_c = Vector2::Zero(Nc2, 2);
    D2.resize(Nc2, Nc2);
    D1x.resize(Nc2, Nc2);
    D1y.resize(Nc2, Nc2);

    std::cout << "initialising variables" << std::endl;

    if (Dc * dt * Nc2 > 0.25) {
        throw std::invalid_argument("timestep condition failed");
    }

    std::vector<Eigen::Triplet<double>> coefficients;

    // fill First derivatives in x
    const double coeff = 1.0 / (2.0 * hc);

    for (int j = 0; j < Nc; ++j) {
        for (int i = 0; i < Nc; ++i) {
            const int id = i * Nc + j;
            insertCoefficient(id, i - 1, j, -coeff, Nc, coefficients);
            insertCoefficient(id, i + 1, j, coeff, Nc, coefficients);
        }
    }
    D1x.setFromTriplets(coefficients.begin(), coefficients.end());

    // fill First derivatives in y
    coefficients.clear();
    for (int j = 0; j < Nc; ++j) {
        for (int i = 0; i < Nc; ++i) {
            int id = i * Nc + j;
            insertCoefficient(id, i, j - 1, -coeff, Nc, coefficients);
            insertCoefficient(id, i, j + 1, coeff, Nc, coefficients);
        }
    }
    D1y.setFromTriplets(coefficients.begin(), coefficients.end());

    // fill Laplacian
    const double a = 1.0 - 4 * Dc * dt / std::pow(hc, 2);
    const double b = Dc * dt / std::pow(hc, 2);

    coefficients.clear();
    for (int j = 0; j < Nc; ++j) {
        for (int i = 0; i < Nc; ++i) {
            const int id = i * Nc + j;
            insertCoefficient(id, i - 1, j, b, Nc, coefficients);
            insertCoefficient(id, i + 1, j, b, Nc, coefficients);
            insertCoefficient(id, i, j - 1, b, Nc, coefficients);
            insertCoefficient(id, i, j + 1, b, Nc, coefficients);
            insertCoefficient(id, i, j, a, Nc, coefficients);
        }
    }
    D2.setFromTriplets(coefficients.begin(), coefficients.end());

    // initial concentration
    auto cinit = [&](const double x, const double y) { return x; };

    const double h = 1.0 / Nc;
    for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
            c(i * Nc + j) = cinit(i * hc, j * hc);
        }
    }
    c = c0 * Nc2 * c / c.sum();

    // create particles and set seed, uncorrelated between samples
    particles.resize(N);
    particles.set_seed(N * sample);

    // [initial cell distribution
    std::uniform_real_distribution<double> uniform(min[0], max[0]);
    std::normal_distribution<double> normal(0, 0.1 * (max[0] - min[0]));
    for (size_t i = 0; i < N; ++i) {
        get<type>(particles)[i] = i < Na;
        auto &gen = get<generator>(particles)[i];
        if (get<type>(particles)[i]) {
            get<position>(particles)[i] = vdouble2(normal(gen), normal(gen));
        } else {
            get<position>(particles)[i] = vdouble2(uniform(gen), uniform(gen));
        }
    }
    if (periodic) {
        for (size_t i = 0; i < N; ++i) {
            auto &gen = get<generator>(particles)[i];
            get<position>(particles)[i] = vdouble2(uniform(gen), uniform(gen));
        }
    }
    // ]

    // [initialise starting variable
    for (size_t i = 0; i < N; ++i) {
        get<starting>(particles)[i] = get<position>(particles)[i];
    }
    // ]

    // [initialise neighbour search
    particles.init_neighbour_search(min, max, vbool2::Constant(periodic));
    // ]

    std::cout << "finished initialising variables. have " << particles.size()
              << " points" << std::endl;
}

void Simulation::integrate(const double time) {
    std::cout << "integrating for " << time << std::endl;
    std::cout << "inter  = " << inter << std::endl;

    const double timesteps = time / dt;

    // [kernel function
    const double kernel_scale1 = 1.0 / (2.0 * PI * std::pow(epsilon, 2));
    const double kernel_scale2 = 1.0 / (2.0 * std::pow(epsilon, 2));
    auto Kbw = [&](const vdouble2 &dx) {
        return kernel_scale1 * std::exp(-dx.squaredNorm() * kernel_scale2);
    };
    // ]

    for (int ts = 0; ts < timesteps; ++ts) {
        std::cout << ".";

        // [chemical gradient calculation
        grad_c.col(0) = chi * D1x * c;  // drift in x-direction
        grad_c.col(1) = chi * D1y * c;  // drift in y-direction
        // ]

        std::uniform_real_distribution<double> uniform;
        std::normal_distribution<double> normal;
        for (Particles_t::reference i : particles) {
            // initialise next position to the current position
            get<next_position>(i) = get<position>(i);

            // [chemical gradient evaluation
            if (get<type>(i)) {
                get<drift>(i) = vdouble2(0, 0);
                get<conc>(i) = 0;
            } else {
                auto &x = get<position>(i);

                const vint2 ind = Aboria::floor((x - min) / hc);

                // linear index to access fx, fy
                const int ind_linear = ind[0] * Nc + ind[1];

                // interpolate gradient
                const vdouble2 x_low = ind * hc + min;
                get<drift>(i) = vdouble2(
                    grad_c(ind_linear, 0) +
                        (grad_c(ind_linear + Nc, 0) - grad_c(ind_linear, 0)) *
                            (x[0] - x_low[0]) * Nc,
                    grad_c(ind_linear, 1) +
                        (grad_c(ind_linear + 1, 1) - grad_c(ind_linear, 1)) *
                            (x[1] - x_low[1]) * Nc);

                get<conc>(i) = c[ind_linear] +
                               (c[ind_linear + Nc] - c[ind_linear]) *
                                   (x[0] - x_low[0]) * Nc +
                               (c[ind_linear + 1] - c[ind_linear]) *
                                   (x[1] - x_low[1]) * Nc;
            }
            // ]

            // [cell - cell interactions
            for (auto j = euclidean_search(particles.get_query(),
                                           get<position>(i), cutoff);
                 j != false; ++j) {
                const double r = j.dx().norm();
                if (r > 0.0) {
                    get<next_position>(i) += -inter * (dt / epsilon) *
                                             std::exp(-r / epsilon) *
                                             (j.dx() / r);
                }
            }
            // ]

            // [brownian diffusion and drift
            auto &gen = get<generator>(i);
            const auto D = get<type>(i) ? Da : Db;
            get<next_position>(i) +=
                dt * get<drift>(i) +
                std::sqrt(2 * D * dt) * vdouble2(normal(gen), normal(gen));
            // ]

            // [reactions between cell types
            const double reaction_propensity =
                (get<type>(i) ? ra : rb * get<conc>(i)) * dt;
            get<type>(i) ^= uniform(gen) < reaction_propensity;
            // ]

            if (!periodic) {
                // [boundary conditions - no flux
                for (size_t d = 0; d < 2; ++d) {
                    if (get<next_position>(i)[d] < min[d]) {
                        get<next_position>(i)[d] =
                            -L - get<next_position>(i)[d];
                    } else if (get<next_position>(i)[d] > max[d]) {
                        get<next_position>(i)[d] = L - get<next_position>(i)[d];
                    }
                }
                // ]
            } else {
                // [boundary conditions - periodic
                for (size_t d = 0; d < 2; ++d) {
                    if (get<next_position>(i)[d] < min[d]) {
                        get<starting>(i)[d] += L;
                    } else if (get<next_position>(i)[d] >= max[d]) {
                        get<starting>(i)[d] -= L;
                    }
                }
                // ]
            }
        }

        // [update particles positions
        for (Particles_t::reference i : particles) {
            get<position>(i) = get<next_position>(i);
        }
        particles.update_positions();
        // ]

        // [cell kernel density evaluation
        rhoalpha.setZero();
        rhobeta.setZero();
        for (int k = 0; k < Nc; ++k) {
            for (int l = 0; l < Nc; ++l) {
                const vdouble2 rgrid = min + vint2(k, l) * hc;
                const int ind_linear = k * Nc + l;
                for (auto i =
                         euclidean_search(particles.get_query(), rgrid, cutoff);
                     i != false; ++i) {
                    if (get<type>(*i)) {
                        rhoalpha(ind_linear) += Kbw(i.dx());
                    } else {
                        rhobeta(ind_linear) += Kbw(i.dx());
                    }
                }
            }
        }
        // ]

        // [chemical concentration update
        c = D2 * c +
            dt * (ka * rhoalpha - kb * c.cwiseProduct(rhobeta) - gam * c);
        // ]
    }
    std::cout << "finished  integrating" << std::endl;
}
