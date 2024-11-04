from directories import home
from matplotlib import pyplot as plt
import json
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import pandas as pd
from trajectory.exp_types import exit_size_HIT, slit_distance
from trajectory.humans import dfs_human
from ConfigSpace.states import min_transitions, reduce_to_state_series

results_folder = home + '\\Analysis\\results\\'

colors = {'Large C': '#F700FF',  # '#8931EF',
          'Large NC': '#00FF13',  # '#cfa9fc',
          'Large': '#cfa9fc',
          'Medium C': '#ad00b3',  # '#00890a',
          'Medium NC': '#00890a',  # '#fab76e',
          'Small': '#000000',
          'all': '#009BFF',
          '': '#009BFF',
          'XL': '#9400D3',
          'L': '#0000FF',
          'M': '#00FF00',
          'S (> 1)': '#FF7F00',
          'S': '#FF7F00',
          'XS': '#FF0000',
          'Single (1)': 'red',
          'Small sim': '#d41e1e',
          'Medium sim': '#61d41e',
          'Large sim': '#1ed4d1',
          'ant': '#fc0000',
          'human': '#000000',
          'DFSRandom': '#40e0d0',
          'DFSSingle': '#7fff00',
          'NC': '#00FF13',
          'C': '#F700FF',
          'gillespie': '#0000FF',
          'randomMemberSim': '#620066',
          'majoritySim': '#00B20D',
          'simC': '#91413b',
          'simNC': '#1300ff',
          'Single': 'red',
          'XL ant': '#9400D3',
          'S (> 1) ant': '#FF7F00',
          'Single (1) ant': 'red',
          ' human': '#009BFF',
          'L ant': '#00FF00',
          'M ant': '#FFD700',
          'SL ant': '#ff9b00',
          'XS ant': '#FF0000',
          }

plt.rcParams.update({'font.size': 8, 'font.family': 'Arial'})

df_human = pd.read_excel(home + '\\lists_of_experiments\\df_human.xlsx')
df_ant_SPT = pd.read_excel(home + '\\lists_of_experiments\\df_ant_SPT.xlsx')
df_ant_HIT = pd.read_excel(home + '\\lists_of_experiments\\df_ant_HIT.xlsx')

with open(home + '\\ConfigSpace\\time_series_human.json', 'r') as json_file:
    time_series_human = json.load(json_file)
    json_file.close()

sizes_per_solver = {'ant': ['S (> 1)', 'M', 'L', 'XL'],
                    'sim': ['S', 'M', 'L', 'XL'],
                    'human': ['Small', 'Medium NC', 'Medium C', 'Large NC', 'Large C'],
                    }

label_dict = {'XL ant': 'large ant group (exp)',
              'S (> 1) ant': 'small ant group (exp)',
              'M ant': 'medium ant group (exp)',
              'L ant': 'medium-large ant group (exp)',
              'Single (1) ant': 'single ant (exp)',
              'XL ant_SPT': 'large ant group (exp)',
              'S (> 1) ant_SPT': 'small ant group (exp)',
              'M ant_SPT': 'medium ant group (exp)',
              'L ant_SPT': 'medium-large ant group (exp)',
              'Single (1) ant_SPT': 'single ant (exp)',
              'XL ant_HIT': 'XL ant group (exp)',
              'XS ant_HIT': 'XS ant group (exp)',
              'S (> 1) ant_HIT': 'S ant group (exp)',
              'M ant_HIT': 'M ant group (exp)',
              'L ant_HIT': 'L ant group (exp)',
              'SL ant_HIT': 'SL ant (exp)',
              'XL gillespie_2024_01_29_no_skirt': 'large ant group (sim, no skirt)',
              'S (> 1) gillespie_2024_01_29_no_skirt': 'small ant group (sim, no skirt)',
              'XL gillespie_2024_01_29_with_skirt': 'large ant group (sim, with skirt)',
              'S (> 1) gillespie_2024_01_29_with_skirt': 'small ant group (sim, with skirt)',
              'XL gillespie_2024_02_15_with_skirt': 'large ant group (sim, with skirt)',
              'XL gillespie_2024_02_21_with_skirt': 'large ant group (sim, with skirt)',
              'S (> 1) gillespie_2024_02_15_with_skirt': 'small ant group (sim, with skirt)',
              'S (> 1) gillespie_2024_02_21_with_skirt': 'small ant group (sim, with skirt)',
              ' human_SPT': 'human solvers',
              'Small': 'single human (exp)',
              'NC': 'human group, \nno communication (exp)',
              'C': 'human group, \ncommunication (exp)',
              'randomMemberSim': 'solverR (sim)',
              'majoritySim': 'solverM (sim)',
              'simNC': 'simNC',
              'simC': 'simC',
              'Large C': 'large human group, \ncommunication (exp)',
              'Large NC': 'large human group, \nno communication (exp)',
              'Medium C': 'medium human group, \ncommunication (exp)',
              'Medium NC': 'medium human group, \nno communication (exp)',
              }

minimal = {
    'ant': {'XL': 64.5, 'L': 64.5 / 2, 'M': 64.5 / 4, 'S': 64.5 / 8, 'S (> 1)': 64.5 / 8, 'Single (1)': 64.5 / 8},
    'gillespie': {'XL': 64.5, 'L': 64.5 / 2, 'M': 64.5 / 4, 'S': 64.5 / 8, 'S (> 1)': 64.5 / 8, 'Single (1)': 64.5 / 8},
    'human': {'Large': 46.813, 'Medium': 46.813 / 2, 'Small Far': 46.813 / 4, 'Small Near': 46.813 / 4}}

solver_step = {'human': 0.05, 'ant': 1, 'sim': 1, 'pheidole': 1, 'DFSRandom': 0.05, 'DFSSingle': 0.05, 'gillespie': 1,
               'randomMemberSim': 0.05, 'majoritySim': 0.05, 'simNC': 0.05, 'simC': 0.05}
linestyle_solver = {'ant': '-', 'sim': (0, (5, 5)), 'human': '-', 'DFSRandom': '-', 'DFSSingle': '-',
                    'gillespie': (0, (5, 5)), 'randomMemberSim': (0, (5, 5)),
                    'majoritySim': (0, (5, 5)), 'simNC': (0, (5, 5)),
                    'simC': (0, (5, 5))}
marker_solver = {'ant': '*', 'sim': '.', 'human': '*', 'DFSRandom': '.', 'DFSSingle': '.', 'gillespie': '.'}


def get_size_groups(df, solver, reduced=False) -> dict:
    if solver in ['ant', 'sim', 'gillespie']:
        df['size'] = df['size'].replace('S', 'S (> 1)')
        if 'SL' in df['size'].unique():  # 'HIT'
            d = {size: df[df['size'] == size] for size in ['L', 'M', 'S (> 1)', 'SL', 'XL', 'XS']}
            d = {size: df_size for size, df_size in d.items() if len(df_size) > 0}
            df_ants_reduced = {'XL': d['XL'], 'SL': d['SL'], 'L': d['L'], 'M': d['M'], 'S (> 1)': d['S (> 1)'],
                               'XS': d['XS']}
            return df_ants_reduced

        else:
            if reduced:
                d = {size: df[df['size'] == size] for size in ['XL', 'S (> 1)', 'Single (1)', ]}
                d = {size: df_size for size, df_size in d.items() if len(df_size) > 0}
                df_ants_reduced = {'XL': d['XL'], 'S (> 1)': d['S (> 1)'], 'Single (1)': d['Single (1)']}
                return df_ants_reduced
            else:
                d = {size: df[df['size'] == size] for size in ['XL', 'L', 'M', 'S (> 1)', 'Single (1)', ]}
                d = {size: df_size for size, df_size in d.items() if len(df_size) > 0}
                df_ants_reduced = {'XL': d['XL'], 'L': d['L'], 'M': d['M'], 'S (> 1)': d['S (> 1)'],
                                   'Single (1)': d['Single (1)']}
                return df_ants_reduced

    elif solver == 'human':
        if reduced:
            dfs_human_reduced = {'Small': dfs_human['Small'],
                                 'NC': pd.concat([dfs_human['Medium NC'], dfs_human['Large NC']]),
                                 'C': pd.concat([dfs_human['Medium C'], dfs_human['Large C']])}
            return dfs_human_reduced
        return dfs_human


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


def expcdf(x, l, x_0):
    return 1 - np.exp(-l * (x - x_0))


def plot_CDFs_path_length(dict_solver_df, reduced=False, color_dict=None, fitted=False, smoothed=True,
                          single=True, legend=''):

    fig_CDF, ax = plt.subplots(figsize=(3.5 + 1.5, 2.5))
    for solver_string, df in dict_solver_df.items():
        solver = solver_string.split('_')[0]
        with open(results_folder + solver_string + '_pL_0.25.json', 'r') as f:
            pL = json.load(f)
        df['measure'] = df['filename'].map(pL)

        if 'winner' not in df.columns:
            df['winner'] = df['filename'].map(json.load(open(results_folder + solver_string + '_winner.json', 'r')))

        dfs = get_size_groups(df, solver, reduced=reduced)
        if not single and solver == 'ant' and 'Single (1)' in dfs.keys():
            dfs.pop('Single (1)')

        if solver == 'human':
            dfs = {'': pd.concat(dfs.values())}

        for size, df_size in tqdm(dfs.items()):
            print(size)
            df_individual = df[df['filename'].isin(df_size['filename'])][['filename', 'size', 'winner', 'measure']]
            if size == 'Single (1)':
                with open(results_folder + 'pL_of_Single_glued.json', 'r') as f:
                    measure = json.load(f)
                df_individual = pd.DataFrame.from_dict(measure, orient='index', columns=['measure'])
                df_individual['size'] = 'Single (1)'
                df_individual['winner'] = [False, False, True]

            if solver_string.split('_')[1] == 'HIT':
                df_individual['norm measure'] = df_individual['measure'] / df_individual['size'].map(exit_size_HIT)
                ax.set_xlabel(r'path length [$d_\mathrm{exit}$]')
            else:
                df_individual['norm measure'] = df_individual['measure'] / df_individual['size'].map(slit_distance[solver])
                ax.set_xlabel(r'path length [$d_\mathrm{cor}$]')
            df_individual = df_individual.sort_values(by='norm measure')
            longest_succ_experiment = df_individual[df_individual['winner']]['norm measure'].max()
            df_individual = df_individual[
                (df_individual['norm measure'] > longest_succ_experiment) | (df_individual['winner'])]
            max_x_value = df_individual[~df_individual['winner']]['norm measure'].min()
            if max_x_value is np.nan:
                max_x_value = longest_succ_experiment
            print(size)
            assert not np.isnan(max_x_value), print('max_x_value is nan' + size)
            x_values = np.arange(0, max_x_value + solver_step[solver], step=solver_step[solver])

            y_values = []
            error_bar = []

            for x in x_values:
                suc = df_individual[(df_individual['norm measure'] < x) & (df_individual['winner'])]
                y_values.append(len(suc) / len(df_individual))
                error_bar.append(np.sqrt(y_values[-1] * (1 - y_values[-1]) / len(df_individual)))  # Bernoulli error

            color = color_dict[size + ' ' + solver]
            if size == 'Single (1)' and single:
                ax.errorbar(x_values[-1], y_values[-1],
                            yerr=error_bar[-1],
                            color=color, marker='x',
                            linestyle='', capsize=3, capthick=1, linewidth=1,
                            label=label_dict[size + ' ' + solver])
            else:
                error_bar_top = np.array([y + e for y, e in zip(y_values, error_bar)])
                error_bar_bottom = np.array([y - e for y, e in zip(y_values, error_bar)])
                x_values, y_values = np.array(x_values), np.array(y_values)
                x_values_middle = (x_values[1:] + x_values[:-1]) / 2
                y_values_middle = (y_values[1:] + y_values[:-1]) / 2
                if fitted and 'gillespie' not in solver:
                    popt, pcov = curve_fit(expcdf, x_values_middle, y_values_middle, p0=[1, 0])
                    ax.plot(x_values_middle, expcdf(x_values_middle, *popt),
                            label=size + ' ' + solver_string, color=color, linestyle=linestyle_solver[solver])
                if not smoothed:
                    ax.fill_between(x_values, error_bar_bottom, error_bar_top, color=color, alpha=0.3)
                    ax.step(x_values, y_values,
                            label=size + ' ' + solver_string,  # + '_' + solver + ' : ' + str(len(df_individual)),
                            color=color,
                            linewidth=1,
                            where='post',
                            linestyle=linestyle_solver[solver])
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

                    ax.fill_between(x_values, error_bar_bottom, error_bar_top, color=color, alpha=0.3)
                    ax.plot(x_values, y_values,
                            label=label_dict[size + ' ' + solver_string],
                            color=color,
                            linewidth=1,
                            linestyle=linestyle_solver[solver])
        largest_size = {'human': 'Large', 'ant': 'XL', 'gillespie': 'XL'}[solver]

        if solver_string != 'ant_HIT':
            ax.axvline(minimal[solver][largest_size] / slit_distance[solver][largest_size],
                       color='grey', linestyle=(0, (5, 5)), linewidth=1, alpha=0.5)
            # write 'minimal' vertically next to the line
            ax.text(minimal[solver][largest_size] / slit_distance[solver][largest_size] + 0.1, 0.8, 'minimal',
                    rotation=90, fontsize=8, color='grey')

    ax.set_ylabel('fraction of success')
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    # put legend on right of the plot
    if legend == 'below':
        ax.legend(prop={'size': 7}, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)
    elif legend == 'false':
        pass
    else:
        ax.legend(prop={'size': 7}, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig_CDF, ax


def states_to_attempted_transitions(s_initial: list):
    s = s_initial.copy()
    # replace all 'ab' and 'ac' and make 'a'
    s = [state.replace('b', '').replace('c', '') if state in ['ab', 'ac'] else state for state in s]

    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            s[i + 1] = ''

        if (s[i] == 'b1' or s[i] == 'b2') and s[i + 1] == 'b':
            s[i + 1] = ''

        if (s[i] == 'be1' or s[i] == 'be2') and s[i + 1] == 'b':
            s[i + 1] = ''

        if (s[i] == 'cg') and s[i + 1] == 'c':
            s[i + 1] = ''

        if (s[i] == 'eg') and s[i + 1] == 'e':
            s[i + 1] = ''

        if (s[i] == 'eb') and s[i + 1] == 'e':
            s[i + 1] = ''

    # remove all empty strings
    s = [x for x in s if x != '']
    return s


def plot_CDFs_attempted_transitions(dict_solver_df, reduced=True, smooth=True, ax=None, fig_CDF=None, legend='',
                                    color_dict=colors):
    if ax is None:
        fig_CDF, ax = plt.subplots(figsize=(3.5, 2.5))

    for solver_string, df in dict_solver_df.items():
        solver = solver_string.split('_')[-1]

        if 'states' not in df.columns:
            ss_dict = {filename: reduce_to_state_series(time_series_human[filename], fps) for filename, fps in
                       zip(df['filename'], df['fps'])}
            df['states'] = df['filename'].map(ss_dict)
        elif 'states' in df.columns and type(df['states'].iloc[-1]) == str:
            df['states'] = df['states'].apply(eval)

        df['transitions'] = df['states'].apply(states_to_attempted_transitions)
        df['measure'] = df['transitions'].apply(len)

        # sort by measure
        df = df.sort_values(by='measure')

        dfs = get_size_groups(df, solver, reduced=reduced)

        for size, df_size in tqdm(dfs.items()):
            df_individual = df[df['filename'].isin(df_size['filename'])][['filename', 'size', 'winner', 'measure',
                                                                          'states', 'transitions']]
            with open(results_folder + size + solver_string + '_transitions_review.json', 'w') as f:
                filename_to_transition_dict = {filename: transitions for filename, transitions in
                                               zip(df_individual['filename'], df_individual['transitions'])}
                json.dump(filename_to_transition_dict, f)
                print('saved in ' + results_folder + size + solver_string + '_transitions_review.json')

            df_individual['norm measure'] = df_individual['measure']
            df_individual = df_individual.sort_values(by='norm measure')
            longest_succ_experiment = df_individual[df_individual['winner']]['norm measure'].max()
            df_individual = df_individual[
                (df_individual['norm measure'] > longest_succ_experiment) | (df_individual['winner'])]
            max_x_value = df_individual[~df_individual['winner']]['norm measure'].min()

            if max_x_value is np.nan:
                max_x_value = longest_succ_experiment
            x_values = np.arange(0, max_x_value + 2 * solver_step[solver], step=solver_step[solver])

            error_bar = []
            y_values = []
            for x in x_values:
                suc = df_individual[(df_individual['norm measure'] < x) & (df_individual['winner'])]
                y_values.append(len(suc) / len(df_individual))
                error_bar.append(np.sqrt(y_values[-1] * (1 - y_values[-1]) / len(df_individual)))  # Bernoulli error

            error_bar_top = np.array([y + e for y, e in zip(y_values, error_bar)])
            error_bar_bottom = np.array([y - e for y, e in zip(y_values, error_bar)])
            if not smooth:
                ax.fill_between(x_values, error_bar_top, error_bar_bottom, color=colors[size], alpha=0.3)
                ax.step(x_values, y_values,
                        label=size,  # + '_' + solver + ' : ' + str(len(df_individual)),
                        color=color_dict[size],
                        linewidth=1,
                        where='post',
                        linestyle=linestyle_solver[solver])
            else:
                # choose only the x values and y values that change as opposed to their previous value
                window_size = int(x_values[-1] / 10 / x_values[1])
                if solver == 'human':
                    y_values[-1] = 1

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

                ax.fill_between(x_values, error_bar_bottom, error_bar_top, color=color_dict[size], alpha=0.3)
                ax.plot(x_values, y_values,
                        label=label_dict[size],  # + '_' + solver + ' : ' + str(len(df_individual)),
                        color=color_dict[size],
                        linewidth=1,
                        linestyle=linestyle_solver[solver])

    ax.axvline(len(min_transitions), color='black', linestyle=(0, (5, 5)), linewidth=1, alpha=0.5)
    ax.text(len(min_transitions) - 1.3, 0.5, 'minimal', fontsize=7, verticalalignment='center', alpha=0.5, rotation=90)

    ax.set_xlabel('number of attempted state transitions')
    ax.set_ylim([0, 1])
    if legend == 'below':
        ax.legend(prop={'size': 7}, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=1)
    else:
        ax.legend(prop={'size': 7})
        ax.set_ylabel('fraction of success')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    return fig_CDF, ax


if __name__ == '__main__':
    # first you have to run the following functions from Analysis/path_length.py to plot these:
    # ants_HIT_save_trans_rot()
    # ants_SPT_save_trans_rot()
    # humans_save_trans_rot()

    _, ax = plot_CDFs_path_length({'ant_HIT': df_ant_HIT},
                                  single=False,
                                  reduced=False,
                                  color_dict=colors,
                                  )
    ax.set_xlim([0, 60])
    plt.savefig('Figures\\Fig_CDF_ant_HIT.svg')
    plt.savefig('Figures\\Fig_CDF_ant_HIT.png', dpi=300)

    _, ax = plot_CDFs_path_length({'ant_SPT': df_ant_SPT, 'human_SPT': df_human},
                                  single=True,
                                  reduced=False,
                                  color_dict=colors,
                                  )
    ax.set_xlim([0, 400])
    plt.savefig('Figures\\Fig_CDF_ant_human_SPT.svg')
    plt.savefig('Figures\\Fig_CDF_ant_human_SPT.png', dpi=300)

    _, ax = plot_CDFs_attempted_transitions({'human': df_human}, reduced=False)
    ax.set_xlim([3, 25])
    plt.savefig('Figures\\Fig_CDF_human_transitions.svg')
    plt.savefig('Figures\\Fig_CDF_human_transitions.png')
