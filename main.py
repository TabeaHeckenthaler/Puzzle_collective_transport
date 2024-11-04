from trajectory.get import get
from Analysis.path_length import PathLength


def play_ant_experiment(filename='XL_SPT_4640021_XLSpecialT_1_ants (part 1)'):
    x = get(filename)
    x.play(step=5, )


def play_human_experiment(filename='large_20210726231259_20210726232557'):
    x = get(filename)
    x.load_participants()
    x.participants.load_from_matlab()
    x.play(step=5, )


def calculate_path_length(filename='XL_SPT_4640021_XLSpecialT_1_ants (part 1)'):
    traj = get(filename)
    pL = PathLength(traj)
    print('translational path length', pL.translational_distance())
    print('rotational path length', pL.rotational_distance())
    print('total path length', pL.pL())


def display_configuration_space():
    pass


if __name__ == '__main__':
    # play_ant_experiment()
    # play_human_experiment()
    # calculate_path_length()
    display_configuration_space()
