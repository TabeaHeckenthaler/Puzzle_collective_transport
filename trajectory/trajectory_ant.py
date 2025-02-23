from directories import NewFileName
from copy import deepcopy
import numpy as np
import scipy.io as sio
from os import path
from trajectory.exp_types import load_periodicity
from trajectory.trajectory_general import Trajectory
from Setup.Maze import Maze, Maze_free_space
from PhysicsEngine.Display import Display

length_unit = 'cm'
trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                  path.sep, path.sep)
trackedPheidoleMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Pheidole Shapes Results'.format(path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep)


class Trajectory_ant(Trajectory):
    def __init__(self, size=None, shape=None, solver=None, old_filename=None, free=False, fps=50, winner=bool,
                 x_error=0, y_error=0, angle_error=0, falseTracking=[], filename=None,
                 position=None, angle=None, frames=None, VideoChain=None, tracked_frames=None):
        if filename is None:
            filename = NewFileName(old_filename, solver, size, shape, 'exp')
        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner,
                         position=position, angle=angle, frames=frames, VideoChain=VideoChain)
        self.x_error = x_error
        self.y_error = y_error
        self.angle_error = angle_error
        self.falseTracking = falseTracking
        self.tracked_frames = []
        self.free = free
        self.state = np.empty((1, 1), int)  # I think I can delete this.
        self.tracked_frames = tracked_frames

    def geometry(self) -> tuple:
        """
        I restarted experiments and altered the maze dimensions for the S, M, L and XL SPT.
        I am keeping track of the movies, that have these altered maze dimensions.

        :return: name of the relevant excel file with the correct dimensions.
        """
        if 'L_I_425' in self.filename:  # This was a single day with these dimensions
            return 'MazeDimensions_ant_L_I_425.xlsx', 'LoadDimensions_ant.xlsx'

        if self.shape != 'SPT':
            return 'MazeDimensions_ant.xlsx', 'LoadDimensions_ant.xlsx'

        new_starting_conditions = [str(x) for x in
                                   list(range(46300, 48100, 100))  # 2021, Tabea
                                   + [44200] +  # Udi's camera 2022
                                   list(range(5000, 6000, 10))  # Lena's camera 2022
                                   ]
        if np.any([self.filename.split('_')[2].startswith(new_starting_condition)
                   for new_starting_condition in new_starting_conditions]):
            return 'MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'

        # elif not self.free:
            # print('You are using old dimensions!, and maybe inaccurate LoadDimensions')

        return 'MazeDimensions_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'

    def __add__(self, file2):
        max_distance_for_connecting = {'XS': 0.8, 'S': 0.3, 'M': 0.3, 'L': 0.3, 'SL': 0.3, 'XL': 0.3}
        if not (self.shape == file2.shape) or not (self.size == file2.size):
            raise ValueError('It seems, that these files should not be joined together.... Please break... ')
        file12 = deepcopy(self)

        file12.position = np.vstack((self.position, file2.position))
        per = 2 * np.pi / load_periodicity[file12.shape]
        a0 = np.floor(self.angle[-1] / per) * per + np.mod(file2.angle[1], per)
        file12.angle = np.hstack((self.angle, file2.angle - file2.angle[1] + a0))
        file12.frames = np.hstack((self.frames, file2.frames))
        file12.tracked_frames = file12.tracked_frames + file2.tracked_frames

        if not self.free:
            file12.winner = file2.winner

        file12.VideoChain = self.VideoChain + file2.VideoChain
        file12.falseTracking = self.falseTracking + file2.falseTracking
        return file12

    def old_filenames(self, i):
        old = None
        if i >= len(self.VideoChain):
            return 'No video found (maybe because I extended)'

        if self.shape[1:] == 'ASH':
            if self.VideoChain[i].split('_')[0] + '_' + \
                    self.VideoChain[i].split('_')[1] == self.size + '_' + self.shape:
                old = self.VideoChain[i].replace(
                    self.VideoChain[i].split('_')[0] + '_' + self.VideoChain[i].split('_')[1],
                    self.size + self.shape[1:]) \
                      + '.mat'
            else:
                print('Something strange in x.old_filenames of x = ' + self.filename)
            #     # this is specifically for 'LASH_4160019_LargeLH_1_ants (part 1).mat'...
            #     old = self.VideoChain[i] + '.mat'
        else:
            old = self.VideoChain[i].replace(self.size + '_' + self.shape, self.size + self.shape) + '.mat'
        return old

    def matlabFolder(self):
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T', 'LongI': 'Long I',
                               'SPT': 'Special T', 'T': 'T'}
        if self.solver == 'pheidole':
            return trackedPheidoleMovieDirectory + path.sep + self.size + path.sep + 'Output Data'

        if not self.free:
            return trackedAntMovieDirectory + path.sep + 'Slitted' + path.sep + shape_folder_naming[
                self.shape] + path.sep + self.size + path.sep + 'Output Data'
        if self.free:
            return trackedAntMovieDirectory + path.sep + 'Free' + path.sep + 'Output Data' + path.sep + \
                   shape_folder_naming[self.shape]

    def matlab_loading(self, old_filename: str, address=None):
        """
        old_filename: str of old_filename with .mat extension
        """
        if not (old_filename == 'XLSPT_4280007_XLSpecialT_1_ants (part 3).mat'):
            if address is None:
                address = self.matlabFolder()
            file = sio.loadmat(address + path.sep + old_filename)

            # if 'Direction' not in file.keys():
            #     file['Direction'] = 'R2L'
            #     print('Direction = R2L')
            # file['Direction'] = None

            if self.shape.endswith('ASH') and 'R2L' == file['Direction']:
                if self.shape == 'LASH':
                    self.shape = 'RASH'
                    self.filename.replace('LASH', 'RASH')
                    self.VideoChain = [name.replace('LASH', 'RASH') for name in self.VideoChain]

                else:
                    self.shape = 'LASH'
                    self.filename.replace('RASH', 'LASH')
                    self.VideoChain = [name.replace('RASH', 'LASH') for name in self.VideoChain]

            if self.shape.endswith('ASH') and self.angle_error[0] == 0:
                if self.shape == 'LASH':
                    self.angle_error = [2 * np.pi * 0.11 + self.angle_error[0]]
                if self.shape == 'RASH':
                    self.angle_error = [-2 * np.pi * 0.11 + self.angle_error[
                        0]]  # # For all the Large Asymmetric Hs I had 0.1!!! (I think, this is why I needed the
                    # error in the end_screen... )

                if self.shape == 'LASH' and self.size == 'XL':  # # It seems like the exit walls are a bit
                    # crooked, which messes up the contact tracking
                    self.angle_error = [2 * np.pi * 0.115 + self.angle_error[0]]
                if self.shape == 'RASH' and self.size == 'XL':
                    self.angle_error = [-2 * np.pi * 0.115 + self.angle_error[0]]

            load_center = file['load_center'][:, :]
            load_center[:, 0] = load_center[:, 0] + self.x_error
            load_center[:, 1] = load_center[:, 1] + self.y_error
            self.frames = file['frames'][0]
            self.tracked_frames = [file['frames'][0][0], file['frames'][0][-1]]
            # # Angle accounts for shifts in the angle of the shape.... (manually, by watching the movies)
            shape_orientation = \
                np.matrix.transpose(file['shape_orientation'][:] * np.pi / 180 + self.angle_error)[0]
            #
            # if file['Direction'] == 'R2L':
            #     shape_orientation = (shape_orientation + np.pi) % np.pi

        else:
            import h5py
            with h5py.File(self.matlabFolder() + path.sep + old_filename, 'r') as f:
                load_center = np.matrix.transpose(f['load_center'][:, :])
                load_center[:, 0] = load_center[:, 0] + self.x_error
                load_center[:, 1] = load_center[:, 1] + self.y_error
                self.frames = np.matrix.transpose(f['frames'][:])[0]
                # # Angle accounts for shifts in the angle of the shape.... (manually, by watching the movies)
                shape_orientation = (f['shape_orientation'][:] * np.pi / 180 + self.angle_error[0])[0]

                # # if not('Direction' in file.keys()) and not(self.shape == 'T' and self.size == 'S'):

        if load_center.size == 2:
            self.position = np.array([load_center])
            self.angle = np.array([shape_orientation])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(shape_orientation)

    def communication(self):
        return False

    def averageCarrierNumber(self) -> float:
        self.load_participants()
        return self.participants.averageCarrierNumber()

    def play(self, wait=0, cs=None, step=1, videowriter=False, frames=None, ts=None, geometry=None, bias=None):
        """
        Displays a given trajectory_inheritance (self)
        :Keyword Arguments:
            * *indices_to_coords* (``[int, int]``) --
              starting and ending frame of trajectory_inheritance, which you would like to display
        """
        x = deepcopy(self)

        if x.frames.size == 0:
            x.frames = np.array([fr for fr in range(x.angle.size)])

        if frames is None:
            f1, f2 = 0, -1
        else:
            f1, f2 = frames[0], frames[1]

        x.position, x.angle = x.position[f1:f2:step, :], x.angle[f1:f2:step]
        x.frames = x.frames[f1:f2:step]

        if not x.free:
            my_maze = Maze(x, geometry=geometry)
        else:
            my_maze = Maze_free_space(x)
            x.position[:, 0] = x.position[:, 0] - np.min(x.position[:, 0])
            x.position[:, 1] = x.position[:, 1] - np.min(x.position[:, 1])

        display = Display(x.filename, x.fps, my_maze, wait=wait, cs=cs, videowriter=videowriter, position=x.position,
                          ts=ts, bias=bias)
        return x.run_trj(my_maze, display=display)

    def solving_time(self) -> float:
        if self.size == 'S':
            self.load_participants()
            cC = self.participants.carriers_attached(self.fps)
            return np.sum(cC != 0) / self.fps
        else:
            return self.timer()

