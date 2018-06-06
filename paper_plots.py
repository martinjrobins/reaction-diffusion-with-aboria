# This Python file uses the following encoding: utf-8

# Import the compiled C++ python module `chemo`
# we have built the module in a subdirectory called `build`,
# so need to append this to the current path
import sys
sys.path.append("build")
import chemo

# Import matplotlib and use off-screen rendering
from matplotlib import rc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Import other necessary python libraries
import numpy as np
from math import sqrt, pi
import pickle
import os.path
import multiprocessing as mp


def get_timestep_data(sim):
    """Extract variables from C++ simulation class.

    This function takes a Simulation class and extract the needed variables
    for post-processing and plotting.

    Args:
        sim: an instance of a C++ simulation class

    """

    # get position and type variables from simulation class
    positions0 = sim.get_starting()
    positions = sim.get_positions()
    the_type = sim.get_type()

    # get chemical concentration and number of grid points
    # reshape concentration as a 2d matrix
    c = sim.get_grid_conc()
    Nc = int(sqrt(len(c)))
    c_grid = np.copy(np.reshape(c, (Nc, Nc), order='C'))

    # get alpha particles and histogram them
    alphas = (the_type == 1).reshape(-1)
    n_alphas = np.sum(alphas)
    alpha_hist = np.histogram2d(
        positions[alphas, 0], positions[alphas, 1], bins=20)

    # get alpha particles and histogram them
    betas = (the_type == 0).reshape(-1)
    n_betas = np.sum(betas)
    print('nalpha = ', n_alphas, 'nbeta = ', n_betas, 'sum = ',
          n_alphas + n_betas, 'nparticles =', len(positions))
    beta_hist = np.histogram2d(
        positions[betas, 0], positions[betas, 1], bins=20)

    # calculate Mean Squared Displacement from starting and current positions
    msd = np.sum((positions - positions0)**2) / len(the_type)
    msd_var = np.var((positions - positions0)**2)
    return n_alphas, alpha_hist, n_betas, beta_hist, c_grid, msd, msd_var


def plot_vis(results, filename):
    """Plot average alpha, beta and c histograms/concentrations.

    This takes the results from N simulations, averages them and then generates figure
    1 and 2 in the paper:

    Figure 1:
        (top row) Histograms of the α particle density normalised by N_α.
        (middle row) Histograms of the β particle density normalised by N_β.
        (Bottom row) Chemical concentration c.

    Figure 2:
        plot of N_α, N_β, and N_α + N_β versus time

    Args:
        results: a list of size N of results from N simulations. Each set of results is a list of
                 M output time results that are returned from `get_timestep_data()`
        filename: a base filename used to save the generated plot

    """

    # setup the particle and grid domain extents
    L = 1.0
    grid_extent = (-0.5 * L, 0.5 * L, -0.5 * L, 0.5 * L)
    particles_extent = (-0.5 * L, 0.5 * L, -0.5 * L, 0.5 * L)

    # results contains `nthreads` simulation results
    # each containing `nout` output datasets
    nthreads = len(results)
    nout = len(results[0])

    # get sizes of histograms (alpha and beta are the same) and chemical grid
    hist_size = results[0][0][1][0].shape[0]
    grid_size = results[0][0][4].shape[0]

    # average number of alpha particles and alpha histogram across N simulations
    n_alphas = [reduce(lambda x, y:x + y[j][0], results, 0) / nthreads
                for j in range(nout)]

    alpha_hist = [reduce(lambda x, y:x + y[j][1][0] / y[j][0], results, np.zeros((hist_size, hist_size)))
                  * hist_size**2 / nthreads for j in range(nout)]

    # average number of beta particles and beta histogram across N simulations
    n_betas = [reduce(lambda x, y:x + y[j][2], results, 0) / nthreads
               for j in range(nout)]
    beta_hist = [reduce(lambda x, y:x + y[j][3][0] / y[j][2], results, np.zeros((hist_size, hist_size)))
                 * hist_size**2 / nthreads for j in range(nout)]

    # average chemical grid across N simulations
    c_grid = [reduce(lambda x, y:x + y[j][4], results, np.zeros((grid_size, grid_size))) / nthreads
              for j in range(nout)]

    # average Mean Squared Displacement across N simulations
    msd = [reduce(lambda x, y:x + y[j][5], results, 0) / nthreads
           for j in range(nout)]

    # get output times (constant for all simulations
    times = [results[0][j][7] for j in range(nout)]

    # start plotting results
    fig, axs = plt.subplots(3, nout)

    # get min/max of histograms and concentrations
    ahist_min = np.min(alpha_hist[0])
    ahist_max = np.max(alpha_hist[0])
    bhist_min = np.min(beta_hist[nout - 2])
    bhist_max = np.max(beta_hist[nout - 2])
    c_min = np.min(c_grid[-1])
    c_max = np.max(c_grid[-1])

    # loop through output timesteps and plot results
    for ts in range(nout):

        # show histograms and concentrations as heat maps using imshow
        ahist = axs[0][ts].imshow(
            np.rot90(alpha_hist[ts]), extent=particles_extent,
            vmin=ahist_min, vmax=ahist_max, cmap='hot', interpolation='nearest')

        bhist = axs[1][ts].imshow(
            np.rot90(beta_hist[ts]), extent=particles_extent,
            vmin=bhist_min, vmax=bhist_max, cmap='hot', interpolation='nearest')

        im = axs[2][ts].imshow(
            np.rot90(c_grid[ts]), extent=grid_extent, vmin=c_min,
            vmax=c_max, cmap='hot', interpolation='nearest')

        # set axis labels and ticks for all three rows
        for i in range(3):
            if ts == 0 and i == 0:
                axs[i][ts].set_ylabel(r'$\rho_\alpha$')
            if ts == 0 and i == 1:
                axs[i][ts].set_ylabel(r'$\rho_\beta$')
            if ts == 0 and i == 2:
                axs[i][ts].set_ylabel(r'$c$')
            if i == 0:
                axs[i][ts].set_title('$t=%s$' % round(times[ts], 3))

            if ts > 0:
                axs[i][ts].get_yaxis().set_ticks([])
                axs[i][ts].get_xaxis().set_ticks([])
            else:
                axs[i][ts].get_yaxis().set_ticks([-0.5, 0.0, 0.5])
                axs[i][ts].get_xaxis().set_ticks([-0.5, 0.0, 0.5])

    # adjust axis to make room for rhs colorbars, then insert colorbars
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.20])
    fig.colorbar(
        im, cax=cbar_ax, ticks=np.round(np.linspace(c_min + 0.002, c_max - 0.002, 3), 2))
    beta_cbar_ax = fig.add_axes([0.90, 0.40, 0.02, 0.20])
    fig.colorbar(bhist, cax=beta_cbar_ax,
                 ticks=np.round(np.linspace(bhist_min + 0.003, bhist_max - 0.004, 3), 2))
    alpha_cbar_ax = fig.add_axes([0.90, 0.67, 0.02, 0.20])
    fig.colorbar(ahist, cax=alpha_cbar_ax,
                 ticks=np.round(np.linspace(ahist_min + 0.002, ahist_max - 0.002, 3), 2))

    # save `vis` figure
    plt.savefig(filename + 'vis.pdf')

    # start `nparticles` figure
    nalphas_fig = plt.figure(3, figsize=(6, 4))
    nalphas_fig.clf()

    # plot particle numbers versus time
    plt.plot(times, n_alphas, label=r'$N_{\alpha}$', linestyle=(0, (5, 5)))
    plt.plot(times, n_betas, label=r'$N_{\beta}$', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(
        times, [i + j for i, j in zip(n_alphas, n_betas)], label=r'$N_{\alpha}+N_{\beta}$')

    # add legend, ticks, axis labels etc.
    plt.legend()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    # save `nparticles` figures
    plt.savefig(filename + 'nparticles.pdf')


def plot_msd(all_results):
    """Plot Mean Squared Displacement figure.

    This takes the results from 3 simulations variants, each with N simulations,
    averages over N and then generates figure 3 in the paper:

    Figure 3:
        The Mean Squared Displacement (MSD) was calculated for three different simula-
        tions: (1) N α = 100, N β = 0 with reactions off, (2) N α = 0, N β = 100
        with reactions off, (3) N α = 50, N β = 50, with reactions on

    Args:
        all_results: a list of size 3, each containing N simulation results.
                    Each set of results has M output time datasets that are returned
                    from `get_timestep_data()`

    """
    # set domain extents
    L = 1.0
    grid_extent = (-0.5 * L, 0.5 * L, -0.5 * L, 0.5 * L)
    particles_extent = (-0.5 * L, 0.5 * L, -0.5 * L, 0.5 * L)

    # loop over 3 simulation variants and average over the N simulations
    msd = []
    msd_var = []
    times = []
    for i, results in enumerate(all_results):
        nout = len(results[0])
        nthreads = len(results)

        # average Mean Squared Displacement across N simulations
        msd.append([reduce(lambda x, y: x + y[j][5], results, 0)
                    / nthreads for j in range(nout)])
        msd_var.append([reduce(lambda x, y: x + y[j][6], results, 0)
                        / nthreads for j in range(nout)])

        # get output times (constant for all simulations
        times.append([results[0][j][7] for j in range(nout)])

    # begin `msd` plot
    msd_fig = plt.figure(figsize=(6, 4))
    msd_fig.clf()

    # plot MSD versus time for each variant
    plt.plot(
        times[0], msd[0], label=r'$N_{\alpha}=100$ and $N_{\beta}=0$', linestyle=(0, (5, 5)))
    plt.plot(
        times[1], msd[1], label=r'$N_{\alpha}=0$ and $N_{\beta}=100$', linestyle=(0, (3, 5, 1, 5)))
    plt.plot(times[2], msd[2], label=r'$N_{\alpha}=50$ and $N_{\beta}=50$')

    # axis labels and legend
    plt.ylabel('Mean Squared Displacement')
    plt.xlabel('$t$')
    plt.legend(loc='best')

    # save `msd` plot
    plt.savefig('msd.pdf')


# set the number N of random simulations to run
Nsim_main = 2000
Nsim_variant = 1000


def plot_all(results):
    """Main plotting function, plots all 3 figures in the paper .

    This takes the results from 1 main and 3 simulations variants, each with N simulations,
    and then uses `plot_vis` and `plot_msd` to generate the three figures in the paper

    Args:
        results: a list of size Nsim_main + 3 * Nsim_variant, containing all the datasets
                 generated

    """

    # Divide `results` into the main set of simulations and 3 variants
    main_results = results[:Nsim_main]
    variant1_results = results[Nsim_main:Nsim_main + Nsim_variant]
    variant2_results = results[
        Nsim_main + Nsim_variant:Nsim_main + 2 * Nsim_variant]
    variant3_results = results[
        Nsim_main + 2 * Nsim_variant:]

    # plot results
    plot_vis(main_results, 'main')
    plot_vis(variant1_results, 'variant1')
    plot_vis(variant2_results, 'variant2')
    plot_vis(variant3_results, 'variant3')
    plot_msd(
        [variant1_results, variant2_results, variant3_results])


def run_thread(sim_index):
    """Main simulation function. Takes an index and runs the corresponding simulation.

    All simulations are cached by saving datasets to a pickle file. If that pickle file
    exists then the results are loaded from that rather than re-running simulation

                          0 <= sim_index < Nsim_main              : main simulation
                  Nsim_main <= sim_index < Nsim_main+Nsim_variant : 1st variant simulation
     Nsim_main+Nsim_variant <= sim_index < Nsim_main+2Nsim_variant: 2nd variant simulation
    Nsim_main+2Nsim_variant <= sim_index < Nsim_main+3Nsim_variant: 3rd variant simulation

    Args:
        sim_index: index of simulation to run

    """
    pickle_fn = 'results%04d.pickle' % sim_index
    if sim_index < Nsim_main:
        simulation_type = 0
        i = sim_index
        final_time = 0.05
        nout = 4
    elif sim_index < Nsim_main + Nsim_variant:
        simulation_type = 1
        i = sim_index - Nsim_main
        final_time = 0.5
        nout = 50
    elif sim_index < Nsim_main + 2 * Nsim_variant:
        simulation_type = 2
        i = sim_index - Nsim_main - Nsim_variant
        final_time = 0.5
        nout = 50
    elif sim_index < Nsim_main + 3 * Nsim_variant:
        simulation_type = 3
        i = sim_index - Nsim_main - 2 * Nsim_variant
        final_time = 0.5
        nout = 50

    if not os.path.isfile(pickle_fn):
        sim = chemo.Simulation(i, simulation_type)
        dt = final_time / (nout - 1)

        datas = []
        for t in range(nout):
            # extract data
            datas.append(get_timestep_data(sim) + (t * dt,))

            # integrate
            sim.integrate(dt)
        pickle.dump(datas, open(pickle_fn, 'wb'))
    else:
        datas = pickle.load(open(pickle_fn, 'rb'))
    return datas


# This is the main routine for the script. If the pickle file exists, it loads
# the simulation results and plots the figures for the paper. If not, it runs
# all the simulations needed to generate the results, and saves these to the
# pickle file.
pickle_fn = 'orig_results.pickle'
if not os.path.isfile(pickle_fn):
    p = mp.Pool(mp.cpu_count())
    results = p.map(run_thread, range(Nsim_main + 3 * Nsim_variant))
    pickle.dump(results, open(pickle_fn, 'wb'))
else:
    results = pickle.load(open(pickle_fn, 'rb'))
plot_all(results)
