from directories import home
from os import path
import json

centerOfMass_shift = - 0.08  # shift of the center of mass away from the center of the SpT load. # careful!
ASSYMETRIC_H_SHIFT = 1.22 * 2
SPT_ratio = 2.44 / 4.82
ant_dimensions = ['ant', 'ps_simulation', 'sim', 'gillespie']  # also in Maze.py
load_periodicity = {'H': 2, 'I': 2, 'RASH': 2, 'LASH': 2, 'SPT': 1, 'T': 1}

average_radius_SPT = {'ant': {'S': 0.921492, 'S (> 1)': 0.921492, 'Single (1)': 0.921492, 'M': 1.842981,
                              'L': 3.662930, 'XL': 7.295145},
                      'human': {'Small Far': 1.1979396, 'Small Near': 1.1979396, 'Small': 1.1979396,
                                'Medium': 2.441953, 'Large': 4.8992658},
                      'gillespie': {'S': 0.921492, 'S (> 1)': 0.921492, 'Single (1)': 0.921492, 'M': 1.842981,
                                    'L': 3.662930, 'XL': 7.295145}}

slit_distance = {'ant': {'XL': 5.830000000000002, 'L': 3.1099999999999994, 'M': 1.56, 'S': 0.77, 'S (> 1)': 0.77,
                         'Single (1)': 0.77},
                 'gillespie': {'XL': 5.830000000000002, 'L': 3.1099999999999994, 'M': 1.56, 'S': 0.77, 'S (> 1)': 0.77},
                 'human': {'Small': 1.06, 'Small Near': 1.06, 'Small Far': 1.06, 'Medium': 2.16, 'Large': 4.24}}

arena_height_SPT = {'ant': {'XL': 19.1, 'L': 9.5, 'M': 4.8, 'S': 2.48, 'S (> 1)': 2.48, 'Single (1)': 2.48, },
                    'human': {'Small': 3.3, 'Medium': 6.39, 'Large': 12.57},
                    'gillespie': {'XL': 19.1, 'L': 9.5, 'M': 4.8, 'S': 2.48, 'S (> 1)': 2.48, 'Single (1)': 2.48, }
                    }

exit_size_HIT = {'XL': 4.9, 'SL': 3.675, 'L': 2.45, 'M': 1.225, 'S': 0.6125, 'S (> 1)': 0.6125, 'XS': 0.31}


def average_radii_HIT(size, shape):
    return {'H': 2.9939 * ResizeFactors['ant'][size],
            'I': 2.3292 * ResizeFactors['ant'][size],
            'T': 2.9547 * ResizeFactors['ant'][size],
            }[shape]


exp_types = {'SPT': {'ant': ['XL', 'L', 'M', 'S'],
                     'pheidole': ['XL', 'L', 'M', 'S'],
                     'human': ['Large', 'Medium', 'Small Far', 'Small Near'],
                     'ps_simulation': ['XL', 'L', 'M', 'S', 'Large', 'Medium', 'Small Far', 'Small Near', ''],
                     'humanhand': [''],
                     'gillespie': ['XL', 'L', 'M', 'S', 'XS']},
             'H': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']},
             'I': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']},
             'T': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']}}

solver_geometry = {'ant': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'human': ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                   'humanhand': ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx'),
                   'gillespie': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'pheidole': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   }

with open(path.join(home, 'Setup', 'ResizeFactors.json'), "r") as read_content:
    ResizeFactors = json.load(read_content)

color = {
    'Large C': '#cc3300',
    'Large communication': '#cc3300',
    'Large non_communication': '#ff9966',
    'Large NC': '#ff9966',
    'M (>7) communication': '#339900',
    'M (>7) non_communication': '#99cc33',
    'Small non_communication': '#0086ff',
    'Small': '#0086ff',
    'XL': '#ff00c1',
    'L': '#9600ff',
    'M': '#4900ff',
    'S (> 1)': '#00b8ff',
    'Single (1)': '#00fff9',
}


def is_exp_valid(shape, solver, size):
    error_msg = 'Shape ' + shape + ', Solver ' + solver + ', Size ' + size + ' is not valid.'
    if shape not in exp_types.keys():
        raise ValueError(error_msg)
    if solver not in exp_types[shape].keys():
        raise ValueError(error_msg)
    if size not in exp_types[shape][solver]:
        raise ValueError(error_msg)
