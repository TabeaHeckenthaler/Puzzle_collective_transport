import numpy as np
from itertools import groupby

states = np.array(['ab', 'ac', 'b1', 'b2', 'be1', 'be2', 'b', 'cg', 'c', 'e', 'eb', "eg", 'f', 'h'])

state_transitions = {'ab': ['ac', 'b'],
                     'ac': ['ab', 'c'],
                     'b1': ['b2', 'be1', 'b', 'be2'],
                     'b2': ['b1', 'be2', 'b', 'be1'],
                     'be1': ['b1', 'be2', 'b', 'b2'],
                     'be2': ['b2', 'be1', 'b', 'b1'],
                     'b': ['b1', 'b2', 'be1', 'be2', 'ab'],
                     'cg': ['c', 'e'],
                     'c': ['cg', 'ac', 'e'],
                     'e': ['eb', 'eg', 'c', 'f', 'cg'],
                     'eb': ['e'],
                     'eg': ['e'],
                     'f': ['h', 'e'],
                     'h': ['f', 'i'],
                     'i': ['h']
                     }

min_states = ['ab', 'ac', 'c', 'e', 'f', 'h']
min_transitions = ['a', 'c', 'e', 'f', 'h']


def reduce_to_state_series(ts, min_length) -> list:
    ts = [state for state in ts if state != 'i']
    # end ts with 'h'
    if ts[-1] != 'h':
        ts.append('h')
    states_lengths = [(''.join(ii[0]), len(list(ii[1]))) for ii in groupby([tuple(label) for label in ts])]
    new_list = [states_lengths[0][0]]

    for sl1, sl2 in zip(states_lengths[1:], states_lengths[2:]):
        if sl1[1] < min_length and (sl2[0] in state_transitions[new_list[-1]] or
                             sl2[0] == new_list[-1]):
            pass
        else:
            if (sl1[0] not in state_transitions[new_list[-1]]) and (sl1[0] != new_list[-1]):
                raise ValueError(sl1[0], 'moved to', new_list[-1])
            if sl1[0] != new_list[-1]:
                new_list.append(sl1[0])
    new_list.append(states_lengths[-1][0])
    new_list = [''.join(ii[0]) for ii in groupby([tuple(label) for label in new_list])]
    if new_list[-1] != 'h':
        new_list.append('h')
    return new_list