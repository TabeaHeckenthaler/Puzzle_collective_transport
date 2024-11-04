from directories import NewFileName
from copy import deepcopy
import numpy as np
import scipy.io as sio
from os import path
from general_functions import ranges
from trajectory.trajectory_general import Trajectory
from trajectory.exp_types import load_periodicity
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

    # def __del__(self):
    #     remove(ant_address(self.filename))

    # def velocity(self, smoothed=False, fps=None):
    #     av_rad = Maze(self).average_radius()
    #
    #     if smoothed:
    #         kernel_size = 2 * (self.fps // 2) + 1
    #         position, unwrapped_angle = self.smoothed_pos_angle(self.position, self.angle, int(kernel_size))
    #     else:
    #         position = self.position
    #         unwrapped_angle = ConnectAngle(self.angle, self.shape)
    #     args = (position[:, 0], position[:, 1], unwrapped_angle * av_rad)
    #
    #     if fps is None:
    #         fps = self.fps
    #
    #     return np.column_stack([np.diff(a) for a in args]) * fps

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
            print('It seems, that these files should not be joined together.... Please break... ')
            breakpoint()

        # if abs(self.position[-1, 0] - file2.position[0, 0]) > max_distance_for_connecting[self.size] or \
        #         abs(self.position[-1, 1] - file2.position[0, 1]) > max_distance_for_connecting[self.size]:
        #     print('does not belong together')
            # breakpoint()

        file12 = deepcopy(self)
        # if not hasattr(file2, 'x_error'):  # if for example this is from simulations.
        #     file2.x_error = 0
        #     file2.y_error = 0
        #     file2.angle_error = 0
        #
        # file12.x_error = [self.x_error, file2.x_error]  # these are lists that we want to join together
        # file12.y_error = [self.y_error, file2.y_error]  # these are lists that we want to join together
        # file12.angle_error = [self.angle_error, file2.angle_error]  # these are lists that we want to join together

        file12.position = np.vstack((self.position, file2.position))

        per = 2 * np.pi / periodicity[file12.shape]
        a0 = np.floor(self.angle[-1] / per) * per + np.mod(file2.angle[1], per)
        file12.angle = np.hstack((self.angle, file2.angle - file2.angle[1] + a0))
        file12.frames = np.hstack((self.frames, file2.frames))
        file12.tracked_frames = file12.tracked_frames + file2.tracked_frames

        if not self.free:
            # file12.contact = self.contact + file2.contact  # We are combining two lists here...
            # file12.state = np.hstack((np.squeeze(self.state), np.squeeze(file2.state)))
            file12.winner = file2.winner  # The success of the attempt is determined, by the fact that the last file
            # is either winner or looser.

        file12.VideoChain = self.VideoChain + file2.VideoChain
        # print(file12.VideoChain, sep="\n")

        file12.falseTracking = self.falseTracking + file2.falseTracking

        # Delete the load of filename
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
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
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

        def check_for_false_tracking(x):
            # print('I would like to renew this check_for_false_tracking function')
            max_Vel_trans, max_Vel_angle = {'XS': 4, 'S': 4, 'M': 2, 'L': 2, 'SL': 2, 'XL': 2}, \
                                           {'XS': 10, 'S': 10, 'M': 2, 'L': 2, 'SL': 2, 'XL': 2}
            vel = x.velocity()
            lister = [x_vel or y_vel or ang_vel or isNaN for x_vel, y_vel, ang_vel, isNaN in
                      zip(vel[0, :] > max_Vel_trans[x.size],
                          vel[1, :] > max_Vel_trans[x.size],
                          vel[2, :] > max_Vel_angle[x.size],
                          np.isnan(sum(vel[:]))
                          )]

            m = ranges(lister, 'boolean', scale=x.frames, smallestGap=20, buffer=8)
            # m = ranges(lister, 'boolean', smallestGap = 20, buffer = 4)
            print('False Tracking Regions: ' + str(m))
            return m

        self.falseTracking = [check_for_false_tracking(self)]
        self.falseTracker()
        self.interpolate_over_NaN()

    def falseTracker(self):
        x = self
        from Setup.Load import periodicity
        per = periodicity[self.shape]

        for frames in x.falseTracking[0]:
            frame1, frame2 = max(frames[0], x.frames[0]), min(frames[1], x.frames[-1])
            index1, index2 = np.where(x.frames == frame1)[0][0], np.where(x.frames == frame2)[0][0]

            con_frames = index2 - index1
            x.position[index1: index2] = np.transpose(
                np.array([np.linspace(x.position[index1][0], x.position[index2][0], num=con_frames),
                          np.linspace(x.position[index1][1], x.position[index2][1], num=con_frames)]))

            # Do we cross either angle 0 or 2pi, when we connect through the shortest distance?
            if abs(x.angle[index2] - x.angle[index1]) / (2 * np.pi / per) > 0.7:
                x.angle[index2] = x.angle[index2] + np.round(
                    (x.angle[index1] - x.angle[index2]) / (2 * np.pi / per)) * (2 * np.pi / per)

            # FinalAngle = np.floor(x.angle[index1]/per)*per + np.mod(x.angle[index2], per)
            x.angle[index1: index2] = np.linspace(x.angle[index1], x.angle[index2], num=con_frames)
            for index in range(index1, index2):
                x.angle[index] = np.mod(x.angle[index], 2 * np.pi)

    def load_participants(self, frames: iter = None) -> None:
        from trajectory_inheritance.ants import Ants
        if not hasattr(self, 'participants') or self.participants is None:
            self.participants = Ants(self)
        if abs(len(self.frames)/len(self.participants.frames)-1) > 0.004:
            raise ValueError('The number of frames in the load and the participants do not match')

    def communication(self):
        return False

    def averageCarrierNumber(self) -> float:
        self.load_participants()
        return self.participants.averageCarrierNumber()

    def find_fraction_of_circumference(self, ts=None) -> np.array:
        x = deepcopy(self)
        my_maze = Maze(self)
        display = Display(x.filename, x.fps, my_maze)
        fc = np.zeros(shape=(0, 3))
        i = 0
        while i < len(self.frames):
            display.renew_screen(movie_name=self.filename,
                                 frame_index=str(self.frames[display.i]) + ', state: ' + ts[i])
            if i == 185:
                DEBUG = 1
            self.step(my_maze, i, display=display)
            if display is not None:
                end = display.update_screen(self, i)
                if end:
                    display.end_screen()
                    self.frames = self.frames[:i]
                    break
            fc = np.vstack([fc, display.calc_fraction_of_circumference()])
            i += 1
        if display is not None:
            display.end_screen()
        return fc

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

    def confine_to_new_dimensions(self) -> 'Trajectory':
        x_new = deepcopy(self)
        my_maze = Maze(x_new)
        new_maze = Maze([], size=x_new.size, shape=x_new.shape, solver='ant',
                        geometry=('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))
        shift_in_x = my_maze.slits[0] - new_maze.slits[0]

        x_new.position[:, 0] = np.maximum(np.zeros_like(x_new.position[:, 0]), x_new.position[:, 0] - shift_in_x)
        x_new.position[:, 1] = x_new.position[:, 1] - (my_maze.arena_height - new_maze.arena_height)/2

        x_new.position[:, 1] = np.minimum(np.maximum(np.zeros_like(x_new.position[:, 1]), x_new.position[:, 1]),
                                         new_maze.arena_height)
        return x_new
        # not fully confining the trajectory_inheritance to the new dimensions


    def frame_count_after_solving_time(self, t):
        frames = t * self.fps
        if self.size == 'S':
            self.load_participants()
            cC = self.participants.carriers_attached(self.fps)
            where = np.where(cC)
            if len(where) < frames:
                return np.NaN
            else:
                return where[frames]
        else:
            return frames
#
#
# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#     # filename = 'S_SPT_4800009_SSpecialT_1_ants (part 1)'
#     filename = 'S_SPT_4710014_SSpecialT_1_ants (part 1)'
#     # filename = 'M_SPT_4710005_MSpecialT_1_ants'
#     x = get(filename)
#     v = x.velocity(4)
#     vel_norm = np.linalg.norm(v, axis=0)
#     plt.plot(vel_norm)
#     stuck = x.stuck(vel_norm=vel_norm, v_min=0.005)
#     plt.plot(np.array(stuck).astype(bool) * 0.2, marker='.', linestyle='', markersize=0.2)
#     # x.angle_error = 0
#     # x.save()
#     # print(x.stuck(v_min=0.005))
#     # TODO: fix that ant traj are saved as simulations
#     DEBUG = 1
#
#     # k_on, k_off = {}, {}
#     # experiments = {'XL': 'XL_SPT_4290009_XLSpecialT_2_ants',
#     #                'L': 'L_SPT_4080033_SpecialT_1_ants (part 1)',
#     #                'M': 'M_SPT_4680005_MSpecialT_1_ants',
#     #                'S': 'S_SPT_4800001_SSpecialT_1_ants (part 1)'}
#     #
#     # for size in experiments.keys():
#     #     x = get('L_SPT_4080033_SpecialT_1_ants (part 1)')
#     #     # x.play()
#     #     x = Trajectory_ant(size=x.size, shape=x.shape, old_filename=x.old_filenames(0), free=False, fps=x.fps, winner=x.winner,
#     #                        x_error=0, y_error=0, angle_error=0, falseTracking=x.falseTracking)
