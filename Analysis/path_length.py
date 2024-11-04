from copy import deepcopy
import numpy as np
import os
from trajectory.exp_types import load_periodicity, average_radius_SPT, average_radii_HIT
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from trajectory.get import get
import json
import pandas as pd
from directories import home
from tqdm import tqdm


class PathLength:
    def __init__(self, x):
        self.traj = deepcopy(x)

    def average_radius(self) -> float:
        if self.traj.shape == 'SPT':
            return average_radius_SPT[self.traj.solver][self.traj.size]
        elif self.traj.shape in ['H', 'I', 'T']:
            return average_radii_HIT(self.traj.size, self.traj.shape)
        else:
            raise ValueError('cant find average radius')

    def translational_distance(self, kernel_size=None, smooth=True) -> float:
        position = self.traj.position
        assert position.size > 5, 'Trajectory is too short to calculate translational distance'
        position_filtered = deepcopy(self.traj.position)
        if smooth:
            position_filtered[:, 0] = self.smooth_array(position[:, 0], int(self.traj.fps/4), kernel_size=kernel_size,)
            position_filtered[:, 1] = self.smooth_array(position[:, 1], int(self.traj.fps/4), kernel_size=kernel_size)
        d = np.abs(np.linalg.norm(np.diff(position_filtered, axis=0), axis=1)).sum()
        return d

    def rotational_distance(self, kernel_size=None, smooth=True) -> float:
        angle = self.traj.angle
        assert angle.size > 5, 'Trajectory is too short to calculate rotational distance'
        assert not np.isnan(angle).any(), \
            'Angle array contains NaNs, use unwrapped_angle = self.ConnectAngle(angle, self.traj.shape)'

        p = load_periodicity[self.traj.shape]
        unwrapped_angle = 1 / p * np.unwrap(p * angle)
        if smooth:
            unwrapped_angle = self.smooth_array(unwrapped_angle, self.traj.fps, kernel_size=kernel_size)

        d = np.abs(np.diff(unwrapped_angle)).sum()  # this is NOT multiplied with average radius, yet!!
        return d

    def pL(self, smooth=True, kernel_size=None) -> float:
        x = self.traj.position[:, 0]
        y = self.traj.position[:, 1]
        theta = np.unwrap(self.traj.angle % (2 * np.pi)) * self.average_radius()

        if smooth:
            x = self.smooth_array(x, int(self.traj.fps/4), kernel_size=kernel_size)
            y = self.smooth_array(y, int(self.traj.fps/4), kernel_size=kernel_size)
            theta = self.smooth_array(theta, self.traj.fps, kernel_size=kernel_size)

        coords = np.stack((x, y, theta), axis=1)
        d = np.abs(np.linalg.norm(np.diff(coords, axis=0), axis=1)).sum()
        return d

    @staticmethod
    def smooth_array(array, fps, kernel_size=None):
        if kernel_size is None:
            kernel_size = 8 * (fps // 4) + 1
        a = medfilt(array, kernel_size=kernel_size)
        a1 = gaussian_filter(a, sigma=kernel_size // 5)
        return a1

    @classmethod
    def example_calculation(cls, filename='XL_SPT_4640021_XLSpecialT_1_ants (part 1)'):
        traj = get(filename)
        pL = cls(traj)
        print('translational path length', pL.translational_distance())
        print('rotational path length', pL.rotational_distance())
        print('rotational path length', pL.rotational_distance() * pL.average_radius())
        print('total path length', pL.pL())

    @classmethod
    def save_pL_trans_rot(cls, df, solver_string):
        trans, rot, pathLength = {}, {}, {}
        for filename in tqdm(df['filename'], desc=solver_string):
            traj = get(filename)
            if traj.solver == 'human':
                traj.smooth(2)
            else:
                traj.smooth(0.25)
            pL = cls(traj)
            trans[filename] = pL.translational_distance(smooth=False)
            rot[filename] = pL.rotational_distance(smooth=False)
            pathLength[filename] = pL.pL(smooth=False)
            print(filename, rot[filename], trans[filename], pathLength[filename])

        results_folder = 'results\\'
        with open(results_folder + solver_string + '_trans_0.25.json', 'w') as f:
            json.dump(trans, f)
        with open(results_folder + solver_string + '_rot_0.25.json', 'w') as f:
            json.dump(rot, f)
        with open(results_folder + solver_string + '_pL_0.25.json', 'w') as f:
            json.dump(pathLength, f)


def ants_HIT_save_trans_rot():
    solver_string = 'ant_HIT'
    df_ants = pd.read_excel(os.path.join(home, 'lists_of_experiments', 'df_ant_HIT.xlsx'))
    df_ants = df_ants.loc[df_ants['free'] == 0]
    PathLength.save_pL_trans_rot(df_ants, solver_string)


def ants_SPT_save_trans_rot():
    solver_string = 'ant_SPT'
    df_ants = pd.read_excel(os.path.join(home, 'lists_of_experiments', 'df_ant_SPT.xlsx'))
    df_ants = df_ants.loc[df_ants['free'] == 0]
    PathLength.save_pL_trans_rot(df_ants, solver_string)


def humans_save_trans_rot():
    solver_string = 'human_SPT'
    df_human = pd.read_excel(os.path.join(home, 'lists_of_experiments', 'df_human.xlsx'))
    PathLength.save_pL_trans_rot(df_human, solver_string)


if __name__ == '__main__':
    PathLength.example_calculation()

    # to run through all the experiments.
    # ants_HIT_save_trans_rot()
    # ants_SPT_save_trans_rot()
    # humans_save_trans_rot()
