from trajectory.get import get
from Analysis.path_length import PathLength
from ConfigSpace.ConfigSpace_Maze import ConfigSpace_Maze
from matplotlib import pyplot as plt
import numpy as np
from directories import home


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
    cs = ConfigSpace_Maze('ant', 'XL', 'SPT',
                          ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
    cs.calculate_space()
    cs.save_space()
    cs.load_space()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    cs.plot(fig, ax)

    traj = get('XL_SPT_4640021_XLSpecialT_1_ants (part 1)')
    coords = np.stack([traj.position[:, 0], traj.position[:, 1], traj.angle], axis=1)
    cs.plot_traj_in_fig(coords, ax)
    plt.savefig(home + '\\ConfigSpace\\Figures\\CS_SPT_XL.png', dpi=300, transparent=True)
    plt.close()


if __name__ == '__main__':
    # play_ant_experiment()
    # play_human_experimen`t()
    # calculate_path_length()
    display_configuration_space()
