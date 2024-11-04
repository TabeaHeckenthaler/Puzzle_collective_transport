from directories import dir_gillespie_trajs, mini_work_dir, df_dir
import os
import pickle
import numpy as np
import pandas as pd
from trajectory.trajectory_ant import Trajectory_ant
from trajectory.trajectory_human import Trajectory_human
from trajectory.trajectory_gillespie import Trajectory_gillespie


def find_simulation_address(filename):
    address = None
    size = filename.split('_')[1]
    for sub_dir in os.listdir(dir_gillespie_trajs):
        if os.path.exists(os.path.join(dir_gillespie_trajs, sub_dir, size, filename)) and address is None:
            address = os.path.join(dir_gillespie_trajs, sub_dir, size, filename)
    if address is None:
        raise FileNotFoundError('Could not find file: ' + filename)
    return address


def get_simulation_traj(file, address):
    if type(file) == dict:
        # ['linVel', 'angVel', 'solver'] I dont use
        shape = file['shape']
        size = file['size']
        filename = file['filename']
        fps = file['fps']
        position = file['position']
        angle = file['angle']
        frames = file['frames']
        nAtt = file['nAtt']
        nPullers = file['nPullers']
        Ftot = file['Ftot']
        OrderParam = file['OrderParam']
        winner = file['winner']

    else:
        shape, size, solver, filename, fps, position, angle, linVel, angVel, frames, winner = file
        filename = address.split('\\')[-1]
        nAtt, nPullers, Ftot, OrderParam = None, None, None, None
    traj = Trajectory_gillespie(shape=shape, size=size, filename=filename,
                             position=position, angle=angle % (2 * np.pi), frames=frames, winner=winner, fps=fps,
                             nAtt=nAtt, nPullers=nPullers, Ftot=Ftot, OrderParam=OrderParam)
    return traj


def get(filename):
    """
    Allows the loading of saved trajectory objects.
    :param filename: Name of the trajectory that is supposed to be un-pickled
    :return: trajectory object
    """
    traj = None
    # first check if the file is in the gillespie directory
    if filename.startswith('sim'):
        address = find_simulation_address(filename)
        with open(address, 'rb') as f:
            file = pickle.load(f)
        traj = get_simulation_traj(file, address)
        return traj

    # this is on labs network
    for root, dirs, files in os.walk(mini_work_dir):
        for dir in dirs:
            df = pd.read_excel(df_dir)
            if filename in os.listdir(os.path.join(root, dir)):
                address = os.path.join(root, dir, filename)
                with open(address, 'rb') as f:
                    file = pickle.load(f)

                if 'Ant_Trajectories' in address:
                    assert filename in df['filename'].values, 'I cannot find ' + filename
                    shape, size, solver, filename, fps, position, angle, frames, winner = file
                    traj = Trajectory_ant(size=size, solver=solver, shape=shape, filename=filename, fps=fps,
                                          winner=winner, position=position, angle=angle, frames=frames,
                                          VideoChain=eval(df[(df['filename'] == filename)]['VideoChain'].iloc[0]),
                                          tracked_frames=eval(
                                              df[(df['filename'] == filename)]['tracked_frames'].iloc[0]), )

                elif 'Human_Trajectories' in address:
                    assert filename in df['filename'].values, 'I cannot find ' + filename
                    shape, size, solver, filename, fps, position, angle, frames, winner = file
                    traj = Trajectory_human(size=size, shape=shape, filename=filename, fps=fps,
                                            winner=winner, position=position, angle=angle, frames=frames,
                                            VideoChain=df[(df['filename'] == filename)]['VideoChain'].iloc[0],
                                            tracked_frames=df[(df['filename'] == filename)]['tracked_frames'].iloc[0])

                elif 'Pheidole_Trajectories' in address:
                    assert filename in df['filename'].values, 'I cannot find ' + filename
                    shape, size, solver, filename, fps, position, angle, frames, winner = file
                    traj = Trajectory_ant(size=size, solver=solver, shape=shape, filename=filename, fps=fps,
                                          winner=winner, position=position, angle=angle, frames=frames,
                                          VideoChain=df[(df['filename'] == filename)]['VideoChain'].iloc[0],
                                          tracked_frames=df[(df['filename'] == filename)]['tracked_frames'].iloc[0], )

                elif 'Gillespie' in address:
                    shape, size, solver, filename, fps, position, angle, linVel, angVel, frames, winner = file
                    traj = Trajectory_gillespie(shape=shape, size=size, filename=address.split('\\')[-1],
                                                position=position, angle=angle % (2 * np.pi),
                                                frames=frames, winner=winner, fps=fps)

                else:
                    raise ValueError('I cannot find directory of ' + filename)
                return traj


if __name__ == '__main__':
    filename = 'XL_SPT_4640021_XLSpecialT_1_ants (part 1)'
    x = get(filename)
    x.play(step=5)
