from itertools import product
from tqdm import tqdm
from Analysis.HIT_stats import HIT_statistics, get_exp_df, color_dict, new_sizing, titles
from directories import directory_results
from matplotlib import pyplot as plt
import numpy as np
from trajectory.trajectory_gillespie import Maze, Trajectory_gillespie
from trajectory.get import get
import os
import pandas as pd
import seaborn as sns
import json
from datetime import datetime


from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

shapes = ['LongI', 'I', 'T', 'H']
size = 'L'
forceTotal = 10 # within 0.4 seconds, its at maximal speed and equilibrizes at around 1.6
# forceTotal = 5 # within 0.4 seconds, its at maximal speed and equilibrizes at around 1.25
time_step = 0.2
gravnon_uniform_force_corner_rnd_corner = [(False, False, False), (True, False, False), (True, True, False)]
keys = ['_'.join(map(str, item)) for item in gravnon_uniform_force_corner_rnd_corner]

color_dict_HIT = {
    'True L True': "#2ca02c",
    'True L False': "#1f77b4",
    'True M True': "#98df8a",
    'True M False': "#aec7e8",
    'False L False': "#d62728",
    'False M False': "#ff9896"
}

labels_HIT = {
    'False L False': titles[0],
    'True L False': titles[1],
    'True L True': titles[2],
    # 'True M True': 'weak grav. on rand. corner',
    # 'True M False': 'weak grav. on same corner',
    # 'False M False': 'weak grav. on center'
}

def flatten(t: list):
    if isinstance(t[0], list):
        return [item for sublist in t for item in sublist]
    else:
        return t


class GridIterator:
    def __init__(self, grid: dict, shapes):
        self.grid = grid
        self.shapes = shapes

        self.prob_rnds = np.unique(np.sort([i[0] for i in flatten(list(self.grid.values()))])).tolist()
        self.kappas = np.unique(np.sort([i[1] for i in flatten(list(self.grid.values()))])).tolist()

    def __len__(self):
        i = 0
        for _ in self:
            i += 1
        return i

    # def __iter__(self):
    #     for prob_rnd, kappa in product(self.prob_rnds, self.kappas):
    #         for force_corner, rnd_corner in self.force_corner_rnd_corner:
    #             for shape in self.shapes:
    #                 yield Calibration(prob_rnd, kappa, force_corner, rnd_corner, shape)

    def __iter__(self, desc=''):
        a = []
        for string, parameters in self.grid.items():
            grav_nonuniform, force_corner, rnd_corner = string.split('_')
            for prob_rnd, kappa, color in parameters:
                for shape in self.shapes:
                    a.append([prob_rnd, kappa, grav_nonuniform, force_corner, rnd_corner, shape, color])

        for args in tqdm(a, total=len(a), desc=desc):
            yield Calibration(*args)

    @classmethod
    def best_grid(cls):
        with open(directory_results + '\\best_values.json', 'r') as f:
            best_values = json.load(f)

        grid = {}
        for key in keys:
            grav_nonuniform, force_on_corner, rnd_corner = key.split('_')
            string = f'{grav_nonuniform}_{force_on_corner}_{rnd_corner}'
            kappa = best_values[string]['kappa']
            prob_rnds = best_values[string]['p']
            grid[key] = [(prob_rnds, kappa, None)]
        return cls(grid, shapes)

    @classmethod
    def ant_grids(cls):
        grids = []
        for shape in shapes:
            grav_non = Calibration.similar_to_ants('XL', shape).grav_nonuniform
            f = Calibration.similar_to_ants('XL', shape).force_corner
            r = Calibration.similar_to_ants('XL', shape).rnd_corner
            keys = [f'{grav_non}_{f}_{r}']
            parameters = []
            for size in ['XL', 'M']:
                parameters.append((Calibration.similar_to_ants(size, shape).prob_rnd,
                                   Calibration.similar_to_ants(size, shape).kappa,
                                   None))
            # parameters = [(p, k, None) for (p, k) in product(prob_rnds, kappas)]
            grid = {k: parameters for k in keys}
            grids.append(cls(grid, [shape]))
        return grids

    def get_efficiencies(self, grav_nonuniform, force_corner, rnd_corner, shape):
        # initialize the values and errors
        values_df = pd.DataFrame()
        annot = pd.DataFrame()

        # open a dictionary with self.prob_rnds as keys
        eff = {str(p): {} for p in self.prob_rnds}
        sem = {str(p): {} for p in self.prob_rnds}

        for prob_rnd, kappa in tqdm(product(self.prob_rnds, self.kappas),
                                    desc=f'Calculating efficiencies {force_corner}_{rnd_corner}',
                                    total=len(self.prob_rnds) * len(self.kappas)):
            cal = Calibration(prob_rnd, kappa, grav_nonuniform, force_corner, rnd_corner, shape)
            hit_stats = HIT_statistics(cal.get_df().copy(), results_folder=cal.folder(), solver_string='gillespie_HIT')
            if 'shape' not in hit_stats.df.columns:
                DEBUG = 1

            effs, sems = hit_stats.get_passing_prob(hit_stats.df[hit_stats.df['shape'] == shape])
            eff[str(prob_rnd)][str(kappa)], sem[str(prob_rnd)][str(kappa)] = effs['L'], sems['L']

            # Convert nested dictionary to DataFrame
            values_df = pd.DataFrame(eff).T
            errors_df = pd.DataFrame(sem).T
            annot = values_df.round(1).astype(str) + " ± " + errors_df.round(1).astype(str)

            values_df.index.name = 'p'
            values_df.columns.name = 'kappa'
        return values_df, annot

    def plot_grid_of_efficiencies(self):
        fig, axss = plt.subplots(len(self.grid), 3, figsize=(10, 10), sharex=True, sharey=True)
        for axs, shape in zip(axss, shapes):
            for ax, force_on_corner_rnd_corner_str, title in tqdm(zip(axs, self.grid.keys(), titles),
                                                           desc='Plotting grid of efficiencies',
                                                           total=len(self.grid)):
                grav_nonuniform, force_on_corner, rnd_corner = [eval(s) for s in force_on_corner_rnd_corner_str.split('_')]

                ax.set_title(title + ': ' + shape)
                values_df, annot = self.get_efficiencies(grav_nonuniform, force_on_corner, rnd_corner, shape)

                # Plot heatmap
                sns.heatmap(values_df,
                            annot=annot,
                            cbar=title == titles[-1],
                            fmt="",
                            cmap="viridis",
                            vmin=0,
                            vmax=1,
                            cbar_kws={'label': r'$P_{\mathrm{slit \rightarrow solution}}$'},
                            annot_kws={"size": 7},  # Adjust font size as nee
                            ax=ax)

                # set the axis labels to r$p$ and r$\kappa$
                if ax == axs[0]:
                    ax.set_ylabel(r'$p$')
                else:
                    ax.set_ylabel('')
                if shape == 'H':
                    ax.set_xlabel(r'$\kappa$')
                else:
                    ax.set_xlabel('')

        plt.tight_layout()

    def plot_minimal_grid_of_efficiencies(self, shape='H', same_parameters=False):
        fig, axs = plt.subplots(1, len(self.grid) + 1, figsize=(9, 4), dpi=300)

        df_ant = get_exp_df()
        hit_stats = HIT_statistics(df_ant, results_folder=directory_results + '\\HIT_stats\\')
        ant_passing_probability, _ = hit_stats.get_passing_prob(hit_stats.df[hit_stats.df['shape'] == shape])
        ant_passing_probability = {key: ant_passing_probability[key] for key in ['M', 'L', 'SL', 'XL']}
        values_df = pd.DataFrame(ant_passing_probability, index=[''])
        values_df.index.name = 'group size'
        sns.heatmap(values_df,  fmt="", cbar=False, cmap="viridis", vmin=0, vmax=1,
                    cbar_kws={'label': r'$P_{\mathrm{slit \rightarrow solution}}$'}, ax=axs[0])
        axs[0].set_title('ant groups (exp.)')
        axs[0].set_ylabel('')

        best_values = {}

        # remove x-tick
        axs[0].set_yticks([])
        axs[0].set_xticklabels(
            [new_sizing[size] for size in ['M', 'L', 'SL', 'XL']],
            rotation=45,  # Rotate labels 45 degrees
            ha='right'  # Align labels to the right
        )

        for ax, title, string in zip(axs[1:], titles, self.grid.keys()):
            grav_nonuniform, force_on_corner, rnd_corner = string.split('_')
            ax.set_title(title)
            values_df, annot = self.get_efficiencies(grav_nonuniform, force_on_corner, rnd_corner, shape)

            # Plot heatmap
            sns.heatmap(values_df,
                        # annot=annot,
                        fmt="",
                        cbar=title==titles[-1],
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                        cbar_kws={'label': r'$P_{\mathrm{slit \rightarrow solution}}$'},
                        ax=ax)

            # set the axis labels to r$p$ and r$\kappa$
            ax.set_xlabel(r'$\kappa$')
            ax.set_ylabel('')

            # highlight the field with the highest value in axs[0]
            if same_parameters:
                row_pos = 1
                col_pos = 1
            else:
                max_idx = values_df.stack().idxmax()  # Returns (row_index, column_name)
                row_pos = values_df.index.get_loc(max_idx[0])  # Convert row label to position
                col_pos = values_df.columns.get_loc(max_idx[1])  # Convert column name to position

                # save values in file
                best_values[f'{grav_nonuniform}_{force_on_corner}_{rnd_corner}'] = {'kappa': float(max_idx[1]),
                                                                                             'p': float(max_idx[0])}
            ax.text(col_pos + 0.5, row_pos + 0.5, 'B', color='red', ha='center', va='center', weight='bold', fontsize=20)

            # if ax == axs[-1]:
            #     for size in ['M', 'XL']:
            #         p, kappa = Calibration.similar_to_ants(size, shape)
            #         row_pos = values_df.index.get_loc(str(p))  # Convert row label to position
            #         col_pos = values_df.columns.get_loc(str(kappa))  # Convert column name to position
            #         ax.text(col_pos + 0.5, row_pos + 0.5, 'ants ' + new_sizing[size],
            #                 color='red', ha='center', va='center', weight='bold',
            #                 fontsize=5)

                # DEBUG = 1



        for ax in axs[1:]:
            ax.set_ylabel(r'$p$')
        axs[0].set_xlabel('group size')

        # write (a, b, c) as labels of the subplots
        for i, ax in enumerate(axs):
            ax.text(-0.2, 1.2, chr(97 + i), transform=ax.transAxes,
                    size=12, weight='bold')

        # image_path = os.path.join(directory_results, 'stuck_coordinates.png')
        # image = plt.imread(image_path)
        # image_box = OffsetImage(image, zoom=0.08)  # Adjust zoom to fit your plot
        # annotation_box = AnnotationBbox(image_box, (-0.2, 0.45), frameon=False, xycoords='axes fraction')
        # axs[0].add_artist(annotation_box)

        axs[2].sharex(axs[1])
        axs[3].sharex(axs[1])

        plt.tight_layout()

        if not same_parameters:
            with open(directory_results +'\\best_values.json', 'w') as f:
                json.dump(best_values, f)

    def plot_CDF(self, string='pL', max_time=50):
        fig, axs = plt.subplots(1, 3, figsize=(7, 2), dpi=300, sharey=True)

        df_ant = get_exp_df()
        for size in ['XL', 'M']:
            for shape, ax in zip(shapes, axs):
                # if not (size == 'XL' and shape == 'H'):
                #     continue
                cal_ants = Calibration.similar_to_ants(size, shape)

                hit_stats_ant_sim = HIT_statistics(cal_ants.get_df(),
                                                   results_folder=cal_ants.folder(),
                                                   solver_string='gillespie_HIT')

                hit_stats_ant_sim.CDF_per_attempt(shape,
                                                  ax=ax,
                                                  labels=[size + ' ant sim.'],
                                                  color=cal_ants.color,
                                                  string=string,
                                                  maxx_time=max_time)

                # I want  to write the parameters cal_ants.prob_rnd and cal_ants.kappa in the plot
                ax.text(0.5, {'M': 0.1, 'XL': 0.9}[size],
                        f'p={cal_ants.prob_rnd}, $\kappa$={cal_ants.kappa}',
                        transform=ax.transAxes,
                        size=5, weight='bold', color='red', ha='center', va='center')

                hit_stats = HIT_statistics(df_ant[df_ant['size'] == size].copy(),
                                           results_folder=directory_results + '\\HIT_stats\\',
                                           solver_string='ant_HIT'
                                           )
                hit_stats.CDF_per_attempt(shape, ax=ax,
                                          labels=[size + ' ant exp.'],
                                          color=color_dict[size + ' ant_HIT'],
                                          string=string,
                                          maxx_time=max_time
                                          )
        #
        for cal in tqdm(self, desc='Plotting CDF for ' + string, total=len(self)):
            hit_stats = HIT_statistics(cal.get_df(), results_folder=cal.folder(), solver_string='gillespie_HIT')
            label = str(cal.grav_nonuniform) + ' L ' + str(cal.force_corner)
            for shape, ax in zip(shapes, axs):
                hit_stats.CDF_per_attempt(shape,
                                          ax=ax,
                                          labels=[labels_HIT[label]],
                                          color=cal.color,
                                          string=string,
                                          maxx_time=max_time
                                          )
        HIT_statistics.finish_CDF_subplots(string, axs)

        # # get position of legend in the last subplot
        # axs[-1].legend().remove()
        # image_path = os.path.join(directory_results, 'different_force_attachment_points.png')
        # image = plt.imread(image_path)
        # image_box = OffsetImage(image, zoom=0.075)  # Adjust zoom to fit your plot
        # annotation_box = AnnotationBbox(image_box, (2.15, 0.45), frameon=False, xycoords='axes fraction')
        #
        # # Add the image in place of the legend
        # axs[-1].add_artist(annotation_box)

        # how do I cut off the part that is now white?
        # plt.tight_layout()

    def run_grid_simulation(self, num=10, frameNumber=500, display=False):
        for cal in self.__iter__(desc='Running simulations: '):
            print(str(cal))
            for _ in range(cal.find_valid_sims(), num):
                cal.run_simulation(display=display, frameNumber=frameNumber)


class Calibration:
    def __init__(self, prob_rnd, kappa, grav_nonuniform, force_corner, rnd_corner, shape, color=None):
        assert prob_rnd <= 1
        self.prob_rnd = prob_rnd
        self.kappa = int(kappa)
        self.shape = shape

        if type(grav_nonuniform) == str:
            self.grav_nonuniform = eval(grav_nonuniform)
        else:
            self.grav_nonuniform = grav_nonuniform

        if type(force_corner) == str:
            self.force_corner = eval(force_corner)
        else:
            self.force_corner = force_corner

        if type(rnd_corner) == str:
            self.rnd_corner = eval(rnd_corner)
        else:
            self.rnd_corner = rnd_corner

        if color is None:
            # 'Uniform gravitation':
            if not self.grav_nonuniform:
                self.color = 'gray'
            elif self.force_corner and not self.rnd_corner:
                self.color = 'darkgreen'
            else:
                self.color = 'cyan'
        else:
            self.color = color

    def __str__(self):
        return (f'prob_rnd={self.prob_rnd}, '
                f'kappa={self.kappa}, '
                f'force_corner={self.force_corner}, '
                f'rnd_corner={self.rnd_corner}, {self.shape}')

    @classmethod
    def similar_to_ants(cls, size, shape):
        # load the json file
        with open('min_distance_CDFs.json', 'r') as f:
            min_dist = json.load(f)

        parameters = min_dist[shape][size]
        p = eval(parameters.split('prob_rnd=')[1].split(', ')[0])
        kappa = eval(parameters.split('kappa=')[1].split(', ')[0])
        color = {'XL': '#c89bff', 'M': '#ffd884'}.get(size)
        return cls(p, kappa, True, True, False, shape, color)

    def run_analytics(self):
        df = self.get_df()
        assert len(df) > 0, 'No simulations to analyze in ' + str(self)
        hit_stats = HIT_statistics(df, results_folder=self.folder(), solver_string='gillespie_HIT')

        # find filenames that do not have indices_of_attempt
        hit_stats.indices_of_attempts()
        hit_stats.pL_per_attempt()
        hit_stats.time_per_attempt()
        self.check_validity_of_folder()

    def folder(self):
        cal_str = f'kappa_{self.kappa}_prob_rnd_{self.prob_rnd}'
        folder = os.path.join(directory_results,
                              f'grav_nonuniform_{self.grav_nonuniform}'
                              f'_force_on_corner_{self.force_corner}_'
                              f'rnd_corner_{self.rnd_corner}\\{cal_str}')

        # create a new folder
        if not os.path.isdir(folder):
            folders = [f for f in folder.split('\\')]
            for i in range(6, len(folders) + 1):
                if not os.path.isdir('\\'.join(folders[:i])):
                    os.mkdir('\\'.join(folders[:i]))

        # save empty dataframe
        if not os.path.isfile(os.path.join(folder, 'df_gillespie.xlsx')):
            pd.DataFrame().to_excel(os.path.join(folder, 'df_gillespie.xlsx'))

        return folder

    def get_pickled_filenames(self):
        return [file for file in os.listdir(self.folder()) if not np.any([file.__contains__(key)
                        for key in ['.json', '.png', '.xlsx', '.txt', '.pdf', '.pkl', '.svg']])]

    def update_df(self):
        df = self.get_df()
        new_filenames = self.get_pickled_filenames()
        if len(df) > 0:
            new_filenames = [file for file in new_filenames if not file in list(df['filename'])]

        df = pd.concat([df, pd.DataFrame(new_filenames, columns=['filename'])], axis=0)
        df['size'] = df['filename'].apply(lambda x: x.split('_')[0])
        df['shape'] = df['filename'].apply(lambda x: x.split('_')[1])
        df['winner'] = df['filename'].apply(lambda x: not 'failed' in x)
        df['solver'] = 'Gillespie'

        # avoid the Unnamed: 0 column
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # update the index
        df = df.reset_index(drop=True)
        df.to_excel(os.path.join(self.folder(), 'df_gillespie.xlsx'))

    def delete_sims(self):
        # find all the simulations
        files = [file for file in os.listdir(self.folder())
                 if not np.any([file.__contains__(key)
                                for key in ['.json', '.png', '.xlsx', '.txt', '.pdf', '.pkl', '.svg']])]

        for filename in files:
            file_path = os.path.join(self.folder(), filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Delete the file
                    print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")

    def find_valid_sims(self):
        # num_valid_sims = \
        #     len([file for file in os.listdir(self.folder())
        #      if file.startswith('L_' + self.shape) and not file.__contains__('.json')])

        df = self.get_df()
        if len(df) == 0:
            return 0
        num_valid_sims = len(df[df['shape'] == self.shape])
        return num_valid_sims

    def check_validity_of_folder(self):
        df = self.get_df()

        names = ['gillespie_HIT_pL_per_attempt.json',
                 'gillespie_HIT_pL_not_attempt.json',
                 'gillespie_HIT_time_per_attempt.json',
                 'gillespie_HIT_time_not_attempt.json'
                ]
        for name in names:
            with open(os.path.join(self.folder(), name), 'r') as f:
                file = json.load(f)

            # remove from df all that are not in file.keys
            assert len(set(list(file.keys())) - set(list(df['filename']))) == 0
            assert len(set(list(df['filename'])) - set(list(file.keys()))) == 0

    def get_df(self):
        return pd.read_excel(os.path.join(self.folder(), 'df_gillespie.xlsx'))

    def new_filename(self):
        df = self.get_df()
        if len(df) == 0:
            n = 0
        else:
            df = df[df['shape'] == self.shape]
            if len(df) == 0:
                n = 1
            else:
                n = df['filename'].apply(lambda x: int(x.split('_')[2])).max() + 1

        assert not np.isnan(n)

        return size + '_' + self.shape + '_' + str(n) + '_' + \
               f'kappa_{self.kappa}_prob_rnd_{self.prob_rnd}' + '_' + datetime.now().strftime("%H-%M-%S")

    def run_simulation(self, display=True, frameNumber=5000, save=True):
        x = Trajectory_gillespie(size=size, shape=self.shape,
                                 filename=self.new_filename(),
                                 forceTotal=forceTotal,
                                 time_step=time_step,
                                 prob_rnd=self.prob_rnd,
                                 kappa=self.kappa,
                                 fps=1 / time_step,
                                 grav_nonuniform=self.grav_nonuniform,
                                 force_on_corner=self.force_corner,
                                 rnd_corner=self.rnd_corner)
        x.run_gravitational_simulation(frameNumber=frameNumber,
                                       display=display)
        if not x.is_solved(Maze(size=x.size, shape=x.shape, solver=x.solver, geometry=x.geometry())):
            x.filename = x.filename + '_failed'

        if save:
            x.save(address=self.folder() + '\\' + x.filename)
            self.update_df()


if __name__ == '__main__':
    # set the constant calibration values
    sizes = ['L']

    # what do I want to do...
    # I want to have 50 simulations of every calibration
    prob_rnds = [0.05, 0.2]  #1
    kappas = [1, 5] # , 32
    shapes = ['LongI', 'I', 'T', 'H']

    parameters = [(p, k, None) for (p, k) in product(prob_rnds, kappas)]

    cal = Calibration(0.1, 5, False, False, False,'LongI')
    for _ in range(50):
        cal.run_simulation(display=True, frameNumber=1000, save=False)

    # traj = get('L_I_0_kappa_1_prob_rnd_0.05_10-59-46')
    # traj.play(wait=100)

    grid = {k: parameters for k in keys}
    grid_iterator = GridIterator(grid, shapes)
    grid_iterator.run_grid_simulation(num=2, display=False)
    for cal in grid_iterator:
        cal.run_analytics()
        # cal.delete_sims()

    grid_iterator.plot_grid_of_efficiencies()
    plt.savefig(os.path.join(directory_results, 'heatmap_grid.pdf'))

    grid_iterator.plot_minimal_grid_of_efficiencies()
    plt.savefig(os.path.join(directory_results, 'heatmap_grid_minimal.pdf'))
