from networkx.algorithms.efficiency_measures import efficiency
from patsy.util import iterable

# from Humans_vs_ants_graphs.Fig2 import results_folder
from trajectory.trajectory_gillespie import *
from directories import home, directory_results
import json
import os
from Setup.Load import corners_phis

import pandas as pd
from tqdm import tqdm
from Analysis.path_length import PathLength
from trajectory.get import get
from matplotlib import pyplot as plt
from scipy.ndimage import label

titles = ['Uniform-Gravity (sim.)', 'Non-Uniform Gravity (sim.)', 'Local-Leader (sim.)']

shapes = ['LongI', 'I', 'T', 'H']
min_pL = 0.05
fitting_boundary = 30
sizes = ['XL', 'SL', 'L', 'M'][::-1]
exit_size = {'XL': 4.9, 'SL': 3.675, 'L': 2.45, 'M': 1.225, 'S': 0.6125, 'S (> 1)': 0.6125, 'XS': 0.31}
color_dict = {
    'XL ant_HIT': '#9400D3',  # '#9400D3',
    'SL ant_HIT': '#0000FF',  # '#f1c40f',  # '#D2691E'
    'L ant_HIT': '#00FF00',  # '#00FF00',
    'M ant_HIT': '#FF7F00',  # '#FFD700',
    'S (> 1) ant_HIT': '#FF7F00',  # 'blue',
    'S ant_HIT': '#FF7F00',  # 'blue',
    'XS ant_HIT': 'red',  # 'red',
    'L gillespie_HIT': 'orange',
    'L sim_HIT': 'orange',
}

new_sizing = {'M': 'small',
              'L': 'medium',
              'SL': 'large',
              'XL': 'extra-large',
              'XL ant exp.': 'extra-l. ant group (exp.)',
              'XL ant sim.': 'extra-l. ant group (sim.)',
              'M ant exp.': 'small ant group (exp.)',
              'M ant sim.': 'small ant group (sim.)',
              }
new_sizing.update({v: v for v in titles})

solver_step = {'human': 0.05, 'ant': 1, 'pheidole': 1}
linestyle_solver = {'ant': '-', 'sim': '--', 'human': '-'}

colors = {
    'Solved I': '#4CAF50',  # Green for "Solved I"
    'Give-up I': '#A5D6A7',  # Lighter green for "Give-up I"
    'Solved T': '#2196F3',  # Blue for "Solved T"
    'Give-up T': '#90CAF9',  # Lighter blue for "Give-up T"
    'Solved H': '#FF9800',  # Orange for "Solved H"
    'Give-up H': '#FFCC80',  # Lighter orange for "Give-up H"
    'Solved XL': '#4CAF50',
    'Give-up XL': '#A5D6A7',
    'Solved SL': '#FF5722',
    'Give-up SL': '#FFAB91',
    'Solved L': '#2196F3',
    'Give-up L': '#90CAF9',
    'Solved M': '#FFC107',
    'Give-up M': '#FFE082',
    'Solved S': '#9C27B0',
    'Give-up S': '#CE93D8',
    'Solved XS': '#3F51B5',
    'Give-up XS': '#7986CB'
}

minimal = {'I': 0.8, 'LongI': 5, 'T': 1.7, 'H': 2.5}

def flatten(t: list):
    if isinstance(t[0], list):
        return [item for sublist in t for item in sublist]
    else:
        return t

class HIT_statistics:
    def __init__(self, df, results_folder, solver_string='ant_HIT'):
        self.df = df
        self.results_folder = results_folder
        self.solver_string = solver_string

    def __getattr__(self, filenames):
        return self.df['filename']

    @classmethod
    def attempt_boolean(cls, traj, maze):
        corners, phis = corners_phis(maze)
        corners = [rotate(traj.position[:, 0], traj.position[:, 1], traj.angle, c) for c in corners]
        # plt.scatter([c[0, 0] for c in corners], [c[1, 0] for c in corners])

        during_attempt = np.zeros(traj.position.shape[0], dtype=bool)
        at_least_on_in_exit_zone = np.any([maze.in_exit_zone(corner) for corner in corners], axis=0)
        during_attempt = np.logical_or(during_attempt, at_least_on_in_exit_zone)

        all_corners_passed = np.all([maze.corner_passed_exit(corner) for corner in corners], axis=0)
        end = np.where(all_corners_passed)
        if len(end) == 0 or len(end[0]) == 0:
            end = len(during_attempt)
        else:
            end = end[0][0]
        during_attempt[end:] = False
        return during_attempt

    def load_eff(self, min_pL=min_pL, string='pL'):
        with open(os.path.join(self.results_folder, self.solver_string + f'_{string}_per_attempt.json'), 'r') as f:
            pL_attempt = json.load(f)

        with open(os.path.join(self.results_folder, self.solver_string + f'_{string}_not_attempt.json'), 'r') as f:
            pL_not_attempt = json.load(f)

        successful_pL_attempt = {key: [list(pL_attempt[key].values())[-1]] if 'failed' not in key else []
                                 for key, value in pL_attempt.items() if len(value) > 0}

        successful_pL_attempt = extend_successful(successful_pL_attempt)

        unsuccessful_pL_attempt = {key: list(pL_attempt[key].values())[:-1] if 'failed' not in key
        else list(pL_attempt[key].values())
                                   for key, value in pL_attempt.items() if len(value) > 0}

        unsuccessful_pL_attempt = {key: [l for l in li if l > exit_size[key.split('_')[0]] * min_pL]
                                   for key, li in unsuccessful_pL_attempt.items()}
        pL_not_attempt = {key: list(pL_not_attempt[key].values())[:-1]
                          for key, value in pL_not_attempt.items() if len(value) > 0}
        return successful_pL_attempt, unsuccessful_pL_attempt, pL_not_attempt

    def time_per_attempt(self):
        with open(os.path.join(self.results_folder, 'indices_of_attempts.json'), 'r') as f:
            indices_of_attempts = json.load(f)

        if os.path.exists(os.path.join(self.results_folder, self.solver_string + '_time_per_attempt.json')):
            with open(os.path.join(self.results_folder, self.solver_string + '_time_per_attempt.json'), 'r') as f:
                time_attempt = json.load(f)
            with open(os.path.join(self.results_folder, self.solver_string + '_time_not_attempt.json'), 'r') as f:
                time_not_attempt = json.load(f)
        else:
            time_attempt, time_not_attempt = {}, {}

        missing_filenames = [filename for filename in self.df['filename'] if filename not in time_attempt.keys()]
        smoothing_times_ant = {'XL': 1 / 2, 'SL': 1 / 2, 'L': 1 / 2, 'M': 1 / 2, 'S': 1 / 2, 'XS': 1 / 2}
        for filename in tqdm(missing_filenames, desc=self.solver_string + ' calc time stats'):
            time_attempt[filename] = {}
            time_not_attempt[filename] = {}
            if filename not in indices_of_attempts.keys() or len(indices_of_attempts[filename]) == 0:
                continue
            attempt_inds, not_attempt_inds = indices_of_attempts[filename]

            if self.solver_string == 'ant_HIT':
                traj = get(filename)
                time_step = 1 / traj.fps
                traj.smooth(smoothing_times_ant[traj.size])
                v = traj.velocity()
                x_vel = v[:, 0]
                y_vel = v[:, 1]
                omega = v[:, 2]  # * average_radius_HIT[traj.shape][traj.size]
                speed = np.linalg.norm(np.stack([x_vel, y_vel, omega]), axis=0)
                v_max = np.percentile(speed, 90)
                print(v_max, traj.filename)
            elif self.solver_string == 'gillespie_HIT':
                time_step = 0.2
                v_max = 1  # cm/s
            else:
                raise ValueError('unknown solver_string')

            for i, indices in enumerate(attempt_inds):
                time_attempt[filename][str(indices)] = np.diff(indices)[0] * time_step * v_max

            for i, indices in enumerate(not_attempt_inds):
                time_not_attempt[filename][str(indices)] = np.diff(indices)[0] * time_step * v_max

        assert len(time_attempt) >= len(self.df)
        assert len(time_not_attempt) >= len(self.df)

        with open(os.path.join(self.results_folder, self.solver_string + '_time_per_attempt.json'), 'w') as f:
            json.dump(time_attempt, f)

        with open(os.path.join(self.results_folder, self.solver_string + '_time_not_attempt.json'), 'w') as f:
            json.dump(time_not_attempt, f)

    def pL_per_attempt(self):
        # load indices of attempts
        with open(os.path.join(self.results_folder, 'indices_of_attempts.json'), 'r') as f:
            indices_of_attempts = json.load(f)

        if os.path.exists(os.path.join(self.results_folder, self.solver_string + '_pL_per_attempt.json')):
            with open(os.path.join(self.results_folder, self.solver_string + '_pL_per_attempt.json'), 'r') as f:
                pL_attempt = json.load(f)
            with open(os.path.join(self.results_folder, self.solver_string + '_pL_not_attempt.json'), 'r') as f:
                pL_not_attempt = json.load(f)
        else:
            pL_attempt = {}
            pL_not_attempt = {}

        smoothing_times_ant = {'XL': 1 / 2, 'SL': 1 / 2, 'L': 1 / 2, 'M': 1 / 2, 'S': 1 / 2, 'XS': 1 / 2}
        missing_filenames = [filename for filename in self.df['filename'] if filename not in pL_attempt.keys()]
        for filename in tqdm(missing_filenames, desc=self.solver_string + ' calc path length stats'):
            traj = get(filename)
            pL_attempt[filename] = {}
            pL_not_attempt[filename] = {}
            if traj.solver == 'human':
                traj.smooth(2)
            else:
                traj.smooth(smoothing_times_ant[traj.size])

            if filename not in indices_of_attempts.keys() or len(indices_of_attempts[filename]) == 0:
                continue

            attempt_inds, not_attempt_inds = indices_of_attempts[filename]
            for i, indices in enumerate(attempt_inds):
                traj_slice = traj.cut_off(indices)
                p_l = PathLength(traj_slice)
                pL_attempt[filename][str(indices)] = p_l.pL(HIT=True)

            for i, indices in enumerate(not_attempt_inds):
                traj_slice = traj.cut_off(indices)
                p_l = PathLength(traj_slice)
                pL_not_attempt[filename][str(indices)] = p_l.pL(HIT=True)

        assert len(pL_attempt) >= len(self.df)
        assert len(pL_not_attempt) >= len(self.df)

        with open(os.path.join(self.results_folder,
                               self.solver_string + '_pL_per_attempt.json'), 'w') as f:
            json.dump(pL_attempt, f)

        with open(os.path.join(self.results_folder,
                               self.solver_string + '_pL_not_attempt.json'), 'w') as f:
            json.dump(pL_not_attempt, f)

    def CDF_per_attempt(self, shape, string='pL', ax=None, labels=None, color=None, with_unsucc=False, maxx_time=50):
        successful_attempt, unsuccessful_attempt, not_attempt = self.load_eff(string=string)

        # Plot cumulative distribution with stacked areas
        assert ax is not None, 'need to pass ax'
        assert type(ax) is not iterable, 'this ax is a list'

        sizes = self.df.loc[self.df['shape'] == shape]['size'].unique()
        # sort size
        sizes = sorted(sizes, key=lambda x: ['M', 'L', 'SL', 'XL'].index(x))
        if labels is None:
            labels = sizes

        for ii, (size, label) in enumerate(zip(sizes, labels)):
            sorted_lengths, y_values_solved, y_values_giveup, error_bar_solved, error_bar_giveup = (
                self.get_CDF_per_attempt(successful_attempt, unsuccessful_attempt, size, shape,
                                         maxx_time=maxx_time, string=string))

            # save the solved and give-up values in a json
            with open(os.path.join(self.results_folder, f'{size}_{shape}_CMD_per_attempt.json'), 'w') as f:
                json.dump({'sorted_lengths': list(sorted_lengths),
                           'solved_cumsum': list(y_values_solved),
                           'giveup_cumsum': list(y_values_giveup)}, f)

            # plot errorbars
            if with_unsucc:
                self.plot_errors_CMD_per_attempt(ax, size, sorted_lengths,
                                                 y_values_giveup,
                                                 error_bar_giveup,
                                                 label='unsucc.' + new_sizing[label],
                                                 color=color,
                                                 smoothed=False,
                                                 linestyle='--'
                                                 )

            # plot errorbars
            self.plot_errors_CMD_per_attempt(ax, size, sorted_lengths, y_values_solved, error_bar_solved,
                                             label=new_sizing[label],
                                             color=color,
                                             smoothed=False,)

            # put the title in the plot on the right bottom
            ax.text(0.95, 0.05, f"{shape}-puzzle", transform=ax.transAxes, ha='right', va='bottom')
            ax.set_xlim([0, {'pL': 10, 'time': maxx_time}[string]])
            ax.set_ylim([0, 100])

        ax.vlines(minimal[shape], 0, 100, color='black', linestyle='--')

    @staticmethod
    def get_CDF_per_attempt(successful_attempt, unsuccessful_attempt, size, shape, maxx_time=50, string='time'):
        successful_attempt_shape = {key: value for key, value in successful_attempt.items()
                                    if key.startswith(size + '_' + shape) and 'failed' not in key}
        unsuccessful_attempt_shape = {key: value for key, value in unsuccessful_attempt.items()
                                      if key.startswith(size + '_' + shape)}

        eff = np.concatenate((*successful_attempt_shape.values(),
                              *unsuccessful_attempt_shape.values())) / exit_size[size]

        successful_values = np.concatenate(tuple(successful_attempt_shape.values()))
        unsuccessful_values = np.concatenate(tuple(unsuccessful_attempt_shape.values()))

        # else:
        #     successful_values = []
        #     unsuccessful_values = []
        #
        #     # combine successful and unsuccessful keys
        #     keys = set(successful_attempt_shape.keys()).union(set(unsuccessful_attempt_shape.keys()))
        #
        #     for key in keys:
        #         if key in successful_attempt_shape.keys() and key in unsuccessful_attempt_shape.keys():
        #             successful_values.append(successful_attempt_shape[key] +
        #                                      np.sum(unsuccessful_attempt_shape[key]))
        #         elif key in successful_attempt_shape.keys():
        #             successful_values.append(successful_attempt_shape[key])
        #         else:
        #             unsuccessful_values.append([np.sum(unsuccessful_attempt_shape[key])])
        #
        #     eff = np.concatenate((*successful_values, *unsuccessful_values)) / exit_size[size]

        if len(successful_values) == 0:
            solved_flags = [False] * len(unsuccessful_values)
        else:
            solved_flags = [True] * len(successful_values) + [False] * len(unsuccessful_values)

        if not len(solved_flags) == len(eff):
            print(f'{size}, {shape} have different solved_flags and path_lengths')
            raise ValueError

        # Combine path lengths and solved flags for sorting
        data = sorted(zip(eff, solved_flags), key=lambda x: x[0])
        sorted_lengths, sorted_flags = zip(*data)
        sorted_lengths = np.array(list(sorted_lengths))
        # is sorted_lengths monotonic?
        assert np.all(np.diff(sorted_lengths) >= 0), 'sorted_lengths not sorted'

        # # find the duplicates in sorted_lengths by finding the indices where the difference is 0
        duplicates = np.where(np.diff(sorted_lengths) == 0)[0]
        # add a very small value to duplicates
        for ii, duplicate in enumerate(duplicates):
            sorted_lengths[duplicate] = sorted_lengths[duplicate] + ii * 10 ** -4
        # # remove the duplicates
        # sorted_lengths = np.delete(sorted_lengths, duplicates)

        error_bar_solved = []
        y_values_solved = []
        error_bar_giveup = []
        y_values_giveup = []
        for i, x in enumerate(sorted_lengths):
            y_values_solved.append(np.sum(sorted_flags[:i + 1]) / len(sorted_lengths))
            # standard error of multinomial distribution
            error_bar_solved.append(
                np.sqrt(y_values_solved[-1] * (1 - y_values_solved[-1]) / len(sorted_lengths)))
            y_values_giveup.append(np.sum(np.logical_not(sorted_flags)[:i + 1]) / len(sorted_lengths))
            error_bar_giveup.append(
                np.sqrt(y_values_giveup[-1] * (1 - y_values_giveup[-1]) / len(sorted_lengths)))

        maxx = {'pL': 10, 'time': maxx_time}[string]
        # # cut off where sorted_lengths is larger than maxx
        y_values_solved = np.array(y_values_solved)[sorted_lengths < maxx] * 100
        error_bar_solved = np.array(error_bar_solved)[sorted_lengths < maxx] * 100
        y_values_giveup = np.array(y_values_giveup)[sorted_lengths < maxx] * 100
        error_bar_giveup = np.array(error_bar_giveup)[sorted_lengths < maxx] * 100
        sorted_lengths = np.array(sorted_lengths)[sorted_lengths < maxx]

        sorted_lengths = np.concatenate(([0], sorted_lengths))
        y_values_solved = np.concatenate(([0], y_values_solved))
        y_values_giveup = np.concatenate(([0], y_values_giveup))
        error_bar_solved = np.concatenate(([0], error_bar_solved))
        error_bar_giveup = np.concatenate(([0], error_bar_giveup))

        sorted_lengths = np.concatenate([sorted_lengths, [maxx]])
        y_values_solved = np.concatenate([y_values_solved, [y_values_solved[-1]]])
        y_values_giveup = np.concatenate([y_values_giveup, [y_values_giveup[-1]]])
        error_bar_solved = np.concatenate([error_bar_solved, [error_bar_solved[-1]]])
        error_bar_giveup = np.concatenate([error_bar_giveup, [error_bar_giveup[-1]]])
        return sorted_lengths, y_values_solved, y_values_giveup, error_bar_solved, error_bar_giveup

    @staticmethod
    def finish_CDF_subplots(string, axs, subplot_labels=('a', 'b', 'c'),):
        xlabel = {'pL': r"path length $(d_{exit})$", 'time': 'norm. time $(d_\mathrm{exit})$'}[string]

        axs[len(axs) // 2].set_xlabel(xlabel)
        axs[0].set_ylabel("cum. percentage (%)")

        # put legend on the right side of the axs[-1]
        # Add legend with unique labels
        handles, labels = axs[-1].get_legend_handles_labels()
        unique = dict(zip(labels, handles))  # Remove duplicates
        axs[-1].legend(unique.values(), unique.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=2)
        plt.tight_layout()

        # write a,b,c in the subplots
        for i, label in enumerate(subplot_labels):
            axs[i].text(-0.1, 1.2, label
                        , transform=axs[i].transAxes, fontsize=12, va='top', ha='right', weight='bold')

    def indices_of_attempts(self):
        # open the json file with the indices of attempts
        if os.path.exists(os.path.join(self.results_folder, 'indices_of_attempts.json')):
            with open(os.path.join(self.results_folder, 'indices_of_attempts.json'), 'r') as f:
                indices_of_attempts = json.load(f)
        else:
            indices_of_attempts = {}
        missing_filenames = [filename for filename in self.filenames if filename not in indices_of_attempts.keys()]

        for filename in tqdm(missing_filenames, desc=self.solver_string + ' calc inds of attempts'):
            traj = get(filename)
            maze = Maze(traj)
            if ('41500' in filename or '41600' in filename) and traj.size == 'XS':
                indices_of_attempts[filename] = []
                # traj.position[:, 0] = traj.position[:, 0] - 10
                # traj.position[:, 1] = traj.position[:, 1] - 6.25
            else:
                attempt_boolean = self.attempt_boolean(traj, maze)
                attempt_boolean = ~filter_short_true_series(~attempt_boolean, traj.fps)
                attempt_boolean = filter_short_true_series(attempt_boolean, traj.fps)
                a = find_ranges(attempt_boolean, min_length=1)
                not_a = find_ranges(~attempt_boolean, min_length=1)
                indices_of_attempts[filename] = (a, not_a)
        with open(os.path.join(self.results_folder, 'indices_of_attempts.json'), 'w') as f:
            json.dump(indices_of_attempts, f)

    def get_passing_prob(self, df_shape):
        successful_pL_attempt, unsuccessful_pL_attempt, pL_not_attempt = self.load_eff()

        df = df_shape.copy()
        df['successful'] = df['filename'].map(successful_pL_attempt)
        df['unsuccessful'] = df['filename'].map(unsuccessful_pL_attempt)
        df = df.loc[~df['successful'].isna() & ~df['unsuccessful'].isna()]

        passing_probability = {}
        se = {}

        for size in df['size'].unique():
            df_size = df.loc[(df['size'] == size)]
            unsucc = df_size['unsuccessful'].tolist()
            succ = df_size['successful'].tolist()

            # remove all nans from unsucc and succ
            unsucc = [l for l in unsucc if type(l) == list]
            succ = [l for l in succ if type(l) == list]

            num_attempts = len(flatten(unsucc) + flatten(succ))
            num_passing = len(flatten(succ))

            p = num_passing / num_attempts
            passing_probability[size] = p
            se[size] = np.sqrt(p * (1 - p) / num_attempts)
        return passing_probability, se

    def efficiency_plot(self, string='time'):
        successful_attempt, unsuccessful_attempt, not_attempt = self.load_eff(string=string)

        fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(6, 2.5), sharey=False)
        self.plot_average_pL_per_attempt(axs[0], successful_attempt)
        if string == 'time':
            axs[0].set_ylabel(r'norm. time $(d_\mathrm{exit})$')
        text = axs[0].text(
            0.9, 0.98,  # Position (x, y) in axes coordinates (0=left, 1=right, 0=bottom, 1=top)
            r"$S_\mathrm{b}^\mathrm{succ}$",  # LaTeX formatted string
            transform=axs[0].transAxes,  # Use axis-relative coordinates
            fontsize=12,  # Adjust font size
            verticalalignment='top',  # Align to the top
            horizontalalignment='right'  # Align to the left
        )

        axs[0].set_ylim([0, 15])
        axs[0].set_yticks([0, 5, 10, 15])

        self.plot_average_pL_per_attempt(axs[1], unsuccessful_attempt)
        # write in the top left corner S_\mathrm{b}^\mathrm{unsucc}
        axs[1].text(
            0.9, 0.98,  # Position (x, y) in axes coordinates (0=left, 1=right, 0=bottom, 1=top)
            r"$S_\mathrm{b}^\mathrm{unsucc}$",  # LaTeX formatted string
            transform=axs[1].transAxes,  # Use axis-relative coordinates
            fontsize=12,  # Adjust font size
            verticalalignment='top',  # Align to the top
            horizontalalignment='right'  # Align to the left
        )
        axs[1].set_ylim([0, 8])

        self.plot_average_pL_per_attempt(axs[2], not_attempt)
        # write in the top left corner  S_\mathrm{a}
        axs[2].text(
            0.9, 0.98,  # Position (x, y) in axes coordinates (0=left, 1=right, 0=bottom, 1=top)
            r"$S_\mathrm{a}$",  # LaTeX formatted string
            transform=axs[2].transAxes,  # Use axis-relative coordinates
            fontsize=12,  # Adjust font size
            verticalalignment='top',  # Align to the top
            horizontalalignment='right'  # Align to the left
        )
        # axs[2].set_ylabel(r'$<t \cdot v_\mathrm{max} S_\mathrm{a} [d_\mathrm{exit}]>$')
        axs[2].set_ylim([0, 25])
        # plt.tight_layout()

        # write a, b, c, d in bold next to the subplots
        for i, letter in enumerate(['a', 'b', 'c']):
            axs[i].text(-0.2, 1.05, letter,
                        transform=axs[i].transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        # plt.tight_layout()

        for ax in axs:
            ax.set_xticklabels([new_sizing[size] for size in sizes],
                               rotation=45,  # Rotate labels 45 degrees
                               ha='right'  # Align labels to the right
                               )

    def plot_geometric_distribution(self):
        # plot a boxplot of the average number of attempts for I, T, H in three subfigures
        df = self.df
        assert self.solver_string == 'ant_HIT'
        successful_pL_attempt, unsuccessful_pL_attempt, pL_not_attempt = self.load_eff()
        df['successful'] = df['filename'].map(successful_pL_attempt)
        df['unsuccessful'] = df['filename'].map(unsuccessful_pL_attempt)

        df = df.loc[~df['successful'].isna() & ~df['unsuccessful'].isna()]

        df = df.loc[~df['filename'].str.contains('failed')]
        df['numAttempts'] = (df['successful'] + df['unsuccessful']).map(len)

        fig, axs = plt.subplots(1, 4, figsize=(10, 2), sharey=True, gridspec_kw={'wspace': 0.3})
        # no space between axes

        for i, shape in enumerate(shapes):
            df_shape = df.loc[df['shape'] == shape]
            data = {size: df_shape.loc[df_shape['size'] == size]['numAttempts'].values for size in sizes}

            # plot the distribution of attempts
            for size in sizes:
                color = color_dict[size + ' ant_HIT']

                # plot histogram as a line
                hist, bins = np.histogram([d-1 for d in data[size]],
                                          bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], density=True)
                axs[i].plot((bins[1:] + bins[:-1]) / 2, hist, color=color, label=new_sizing[size],
                            marker='x', linestyle='-')
                # write shape + puzzle in the bottom right corner
            axs[i].text(0.95, 0.85, shape + '-puzzle', transform=axs[i].transAxes, ha='right', va='bottom', fontsize=10)

            axs[i].set_ylim([0, 1])
            axs[i].set_xlim([0, 8])
            # make x ticks at integer values
            axs[i].set_xticks(range(0, 8, 2))
            axs[i].set_yticks([0, 0.5, 1])
        # legend in to middle right
        axs[0].legend(loc='center right', bbox_to_anchor=(1, 0.5), ncol=1)
        axs[1].set_xlabel(r'$N$, number of $(S_\mathrm{slit} \to S_\mathrm{bulk})$ reps.')
        axs[0].set_ylabel('pdf')

        self.plot_passing_probabilty(axs[-1])

        # write a,b,c in bold
        for i, letter in enumerate(['a', 'b', 'c', 'd']):
            axs[i].text(-0.2, 1.2, letter, transform=axs[i].transAxes, fontsize=12, fontweight='bold', va='top')

    def plot_errors_CMD_per_attempt(self, ax, size, x_values, y_values, error_bar, label, color=None, smoothed=False,
                                    linestyle='-'):
        solver = 'ant'
        error_bar_top = np.array([y + e for y, e in zip(y_values, error_bar)])
        error_bar_bottom = np.array([y - e for y, e in zip(y_values, error_bar)])
        x_values, y_values = np.array(x_values), np.array(y_values)

        if color is None:
            color = color_dict[size + ' ' + self.solver_string]

        if not smoothed:
            ax.fill_between(x_values, error_bar_bottom, error_bar_top,
                            color=color,
                            alpha=0.2)
            ax.step(x_values, y_values,
                    label=label,  # + '_' + solver + ' : ' + str(len(df_individual)),
                    color=color,
                    linewidth=1,
                    where='post',
                    # linestyle={'unsucc.': ':', 'succ.': '-'}[label.split(' ')[0]],
                    linestyle=linestyle
                    )

        if smoothed:
            # choose only the x values and y values that change as opposed to their previous value
            window_size = int(x_values[-1] / 10 / x_values[1])

            if y_values[-1] > 0.99:
                # append many 1s to the end of the array and extend x_values
                y_values = np.append(y_values, [1] * window_size)
                x_values = np.append(x_values, np.arange(x_values[-1] + solver_step[solver],
                                                         x_values[-1] + (1 + window_size) * solver_step[solver],
                                                         step=solver_step[solver]))[:len(y_values)]
                error_bar_top = np.append(error_bar_top, [1] * window_size)[:len(y_values)]
                error_bar_bottom = np.append(error_bar_bottom, [1] * window_size)[:len(y_values)]

            _, y_values = smooth_data(x_values, y_values, window_size=window_size)
            _, error_bar_top = smooth_data(x_values, error_bar_top, window_size=window_size)
            _, error_bar_bottom = smooth_data(x_values, error_bar_bottom, window_size=window_size)

            ax.fill_between(x_values, error_bar_bottom, error_bar_top, color=colors[label], alpha=0.3)
            ax.plot(x_values, y_values,
                    label=label,
                    color=colors[label],
                    linewidth=1,
                    linestyle=linestyle_solver[solver])

    def plot_passing_probabilty(self, axs, marker='x'):
        shift = {'H': 0.2, 'I': -0.2, 'LongI': -0.3, 'T': 0}
        for i, shape in enumerate(shapes):
            passing_probability, se = self.get_passing_prob( self.df.loc[self.df['shape'] == shape])
            for ii, size in enumerate(sizes):
                color = color_dict[size + ' ant_HIT']
                axs.errorbar(ii + shift[shape], passing_probability[size],
                             yerr=se[size], color=color, marker=marker)
                # write the shape close to the marker
                letter_shift = -0.2  # if passing_probability[shape][size] > 0.5 else 0.15
                axs.text(ii + shift[shape], passing_probability[size] + letter_shift, shape,
                         color=color, fontsize=8, ha='center', va='bottom')

        # set x markers to sizes
        axs.set_xticks(range(len(sizes)))
        axs.set_xticklabels([new_sizing[s] for s in sizes], rotation=45, ha="right", rotation_mode="anchor")
        axs.set_ylim([0, 1])
        axs.set_ylabel(r'$P_{\mathrm{slit \rightarrow solution}}$')
        axs.set_xlabel('size')

    def plot_average_pL_per_attempt(self, ax, pL):
        # plot a boxplot of the average number of attempts for I, T, H in three subfigures
        self.df['pL'] = self.df['filename'].map(pL)
        sizes = self.df['size'].unique()
        # sort sizes by ['M', 'L', 'SL', 'XL']]
        sizes = sorted(sizes, key=lambda x: ['M', 'L', 'SL', 'XL'].index(x))
        for i, shape in enumerate(shapes):
            data = {}
            for size in sizes:
                df_shape_size = self.df.loc[(self.df['shape'] == shape) & (self.df['size'] == size)]
                k = [di for di in df_shape_size['pL'].values]

                # remove all nans from k
                k = [ki for ki in k if type(ki) == list]
                if np.all([len(ki) == 0 for ki in k]):
                    print('\nno data for ' + size + ' ' + shape + '\n')

                try:
                    data[size] = np.concatenate(k)
                except ValueError:
                    data[size] = []

            # I want an errorbar plot with means
            mean, std, sem = {}, {}, {}
            for j, size in enumerate(sizes):
                if len(data[size]) == 0:
                    mean[size] = np.nan
                    sem[size] = np.nan
                    continue
                else:
                    data[size] = data[size][~np.isnan(data[size])]
                    mean[size] = np.mean(data[size]) / exit_size[size]
                    sem[size] = np.std(data[size]) / np.sqrt(len(data[size])) / exit_size[size]

            marker = {'H': '.', 'I': '.', 'T': '.', 'LongI': '.'}
            shift = {'H': 0.2, 'I': -0.2, 'T': 0, 'LongI': -0.3,}

            for ii, size in enumerate(sizes):
                color = color_dict[size + ' ant_HIT']
                ax.errorbar(ii + shift[shape], mean[size],
                            yerr=sem[size], color=color, marker=marker[shape])
                ax.text(ii + shift[shape], (mean[size] + sem[size]) * 1.05, shape,
                        color=color, fontsize=10, ha='center', va='bottom')
        # set x markers to sizes
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.set_xlim([-0.5, len(sizes) - 0.5])
        ax.set_xlabel('size')

def find_ranges(bool_list, min_length=0, max_length=np.inf) -> list:
    inds = np.array([i for i, x in enumerate(bool_list) if x])
    index_successions_in_a = np.split(inds, np.where(np.diff(inds) != 1)[0] + 1)

    ranges = [[int(ind[0]), int(ind[-1] + 1)] for ind in index_successions_in_a if max_length > len(ind) > min_length]
    return ranges

def extend_successful(succesful):
    for key, value in succesful.items():
        size, shape = key.split('_')[:2]
        for i, v in enumerate(value):
            if v < exit_size[size] * minimal[shape]:
                succesful[key][i] = exit_size[size] * minimal[shape]
    return succesful

def smooth_data(x, y, window_size=3):
    """
    Smooths out data points using a moving average.

    Parameters:
        x (array-like): Array containing x values.
        y (array-like): Array containing y values.
        window_size (int): Size of the moving average window.

    Returns:
        tuple: Tuple containing two arrays: smoothed x values and smoothed y values.
    """
    smoothed_y_values = []

    # Extend the beginning of the array with the first value
    y = np.concatenate(([y[0]] * (window_size - 1), y))
    x = np.concatenate(([x[0] - i * (x[1] - x[0]) for i in range(window_size - 1)], x))

    for i in range(len(y) - window_size + 1):
        # Calculate the moving average for y values
        smoothed_y_values.append(np.mean(y[i:i + window_size]))

    return x[window_size - 1:], np.array(smoothed_y_values)


def rotate(x, y, angle, c):
    return np.array([x + c[0] * np.cos(angle) - c[1] * np.sin(angle),
                     y + c[0] * np.sin(angle) + c[1] * np.cos(angle)])



def filter_short_true_series(array, min_len):
    # Identify connected regions of True values
    labeled_array, num_features = label(array)

    # Create a mask for series that meet the minimum length condition
    mask = np.zeros_like(array, dtype=bool)
    for i in range(1, num_features + 1):  # Iterate over each labeled region
        region = (labeled_array == i)
        if np.sum(region) >= min_len:
            mask |= region  # Add this region to the mask if it meets the condition

    return mask


def get_exp_df():
    df = pd.read_excel(home + '\\DataFrame\\final\\df_ant_HIT.xlsx')
    if 'free' in df.columns:
        df = df.loc[df['free'] == 0]

    df = df.loc[df['size'] != 'XS']
    df = df.loc[df['size'] != 'S']

    remove_filename = ['L_I_4250004_4_ants (part 1)',  # they broke through the Fluon here
                       'SL_H_4130004_2_ants',  # not enough ants to lift the back
                       ]
    df = df.loc[~df['filename'].isin(remove_filename)]
    return df

if __name__ == '__main__':
    # for experiments
    df_ant = get_exp_df()
    results_folder = directory_results + '\\HIT_stats\\'
    hit_stats = HIT_statistics(df_ant, solver_string='ant_HIT', results_folder=results_folder)
    # successful_pL_attempt, unsuccessful_pL_attempt, pL_not_attempt = hit_stats.load_eff()
    # shape = 'H'
    # df_ant = df_ant[df_ant['shape'] == shape]
    # df_ant['successful'] = df_ant['filename'].map(successful_pL_attempt)
    # df_ant['unsuccessful'] = df_ant['filename'].map(unsuccessful_pL_attempt)
    # df_ant = df_ant.loc[~df_ant['successful'].isna() & ~df_ant['unsuccessful'].isna()]

    # passing_probability, se = hit_stats.get_passing_prob(shape=shape)


    # running analytics
    # results_folder = home + '\\Analysis\\Efficiency\\results_double_check\\'
    hit_stats = HIT_statistics(df_ant, results_folder=results_folder, solver_string='ant_HIT')
    # hit_stats.indices_of_attempts()
    # hit_stats.pL_per_attempt()
    # hit_stats.time_per_attempt()

    # plotting analytics
    hit_stats.plot_geometric_distribution()
    plt.savefig(os.path.join(hit_stats.results_folder, 'Fig2_geometric_distribution.pdf'), bbox_inches='tight')
    # #
    # for efficiency_measure in ['time']: #'pL',
    #     hit_stats.efficiency_plot(string=efficiency_measure)
    #     plt.savefig(os.path.join(hit_stats.results_folder, f'Fig2_performance_passing_Prob_{efficiency_measure}.pdf'))

    #     hit_stats.CDF_per_attempt(string=efficiency_measure, maxx_time=20)
    #     plt.savefig(os.path.join(hit_stats.results_folder,
    #                 f'Fig2_performance_CMD_per_attempt_{efficiency_measure}.pdf'),
    #                 bbox_inches='tight')
    # #
