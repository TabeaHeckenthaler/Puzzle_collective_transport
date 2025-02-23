"""
Created on Wed May  6 11:24:09 2020

@author: tabea
"""
import numpy as np
from os import path
import os

import pandas as pd
from general_functions import smooth_array
import pickle
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import json
from trajectory.exp_types import is_exp_valid
from copy import copy
from datetime import datetime
from directories import dirs_exp_trajs, lists_exp_dir, work_dir
from Setup.Maze import Maze
from trajectory.exp_types import load_periodicity
from PhysicsEngine.Display import Display
from directories import directory_results


""" Making Directory Structure """
sizes = {'ant': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'pheidole': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'human': ['Small Far', 'Small Near', 'Medium', 'Large'],
         'humanhand': ''}

solver_geometry = {'ant': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'pheidole': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'human': ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                   'humanhand': ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')}

solvers = ['ant', 'pheidole', 'human', 'humanhand', 'ps_simulation']

length_unit = {'ant': 'cm', 'human': 'm', 'humanhand': 'cm', 'ps_simulation': 'cm'}


def continue_time_dict(name):
    """
    Saves the length of experiments in seconds. I need this because I am missing part of my trajectories.
    :param solver:
    :param shape:
    :return:
    """

    def delta_t(name):
        """
        :return: in seconds
        """
        print(name)
        l = name.split('_')
        name = str(int(l[1]))[-2:]
        start_string = input(name + '  start: ')
        start_string = (start_string.split('.')[0].lstrip('0') or '0') + '.' \
                       + (start_string.split('.')[1].lstrip('0') or '0')

        end_string = input(name + '  end: ')
        end_string = (end_string.split('.')[0].lstrip('0') or '0') + '.' \
                     + (end_string.split('.')[1].lstrip('0') or '0')
        return int((datetime.strptime(end_string, '%H.%M') - datetime.strptime(start_string, '%H.%M')).total_seconds())

    with open('time_dictionary.txt', 'r') as json_file:
        time_dict = json.load(json_file)

    time_dict.update({name: delta_t(name)})
    with open('time_dictionary.txt', 'w') as json_file:
        json.dump(time_dict, json_file)


class Trajectory:
    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool, VideoChain=None,
                 angle=np.empty((1, 1), float), position=np.empty((1, 2), float), frames=np.empty(0, float)):
        is_exp_valid(shape, solver, size)
        self.shape = shape  # shape (maybe this will become name of the maze...) (H, I, T, SPT)
        self.size = size  # size (XL, SL, L, M, S, XS)
        self.solver = solver  # ant, human, sim, humanhand
        self.filename = filename  # filename: shape, size, path length, sim/ants, counter

        if VideoChain is None:
            self.VideoChain = [self.filename]
        else:
            self.VideoChain = VideoChain

        self.fps = fps  # frames per second
        self.position = position  # np.array of x and y positions of the centroid of the shape
        self.angle = angle  # np.array of angles while the shape is moving
        self.frames = frames
        self.winner = winner  # whether the shape crossed the exit
        self.participants = None
        self.ts = None

    def __bool__(self):
        return self.winner

    def __str__(self):
        string = '\n' + self.filename
        return string

    def __add__(self, file2):
        file12 = deepcopy(self)
        file12.position = np.vstack((self.position, file2.position))

        per = 2 * np.pi / load_periodicity[file12.shape]
        a0 = np.floor(self.angle[-1] / per) * per + np.mod(file2.angle[1], per)
        file12.angle = np.hstack((self.angle, file2.angle - file2.angle[1] + a0))
        file12.frames = np.hstack((self.frames, file2.frames))
        if hasattr(self, 'tracked_frames'):
            file12.tracked_frames = file12.tracked_frames + file2.tracked_frames
        if hasattr(self, 'VideoChain'):
            file12.VideoChain = self.VideoChain + file2.VideoChain
            # print(file12.VideoChain, sep="\n")
        if hasattr(self, 'falseTracking'):
            file12.falseTracking = self.falseTracking + file2.falseTracking
        return file12

    def step(self, my_maze, i):
        my_maze.set_configuration(self.position[i], self.angle[i])

    def smooth(self, sec_smooth):
        if self.solver == 'gillespie':
            return
        self.position[:, 0] = smooth_array(self.position[:, 0], int(sec_smooth * self.fps))
        self.position[:, 1] = smooth_array(self.position[:, 1], int(sec_smooth * self.fps))
        # unwrapped_angle = ConnectAngle(self.angle, self.shape)
        unwrapped_angle = np.unwrap(self.angle)
        self.angle = smooth_array(unwrapped_angle, int(sec_smooth * self.fps))

    def interpolate_over_NaN(self):
        if np.any(np.isnan(self.position)) or np.any(np.isnan(self.angle)):
            nan_frames = np.unique(np.append(np.where(np.isnan(self.position))[0], np.where(np.isnan(self.angle))[0]))

            fr = [[nan_frames[0]]]
            for i in range(len(nan_frames) - 1):
                if abs(nan_frames[i] - nan_frames[i + 1]) > 1:
                    fr[-1] = fr[-1] + [nan_frames[i]]
                    fr = fr + [[nan_frames[i + 1]]]
            fr[-1] = fr[-1] + [nan_frames[-1]]
            print('Was NaN...' + str([self.frames[i].tolist() for i in fr]))

        # Some of the files contain NaN values, which mess up the Loading.. lets interpolate over them
        if np.any(np.isnan(self.position)) or np.any(np.isnan(self.angle)):
            for indices in fr:
                if indices[0] < 1:
                    indices[0] = 1
                if indices[1] > self.position.shape[0] - 2:
                    indices[1] = indices[1] - 1
                con_frames = indices[1] - indices[0] + 2
                self.position[indices[0] - 1: indices[1] + 1, :] = np.transpose(np.array(
                    [np.linspace(self.position[indices[0] - 1][0], self.position[indices[1] + 1][0], num=con_frames),
                     np.linspace(self.position[indices[0] - 1][1], self.position[indices[1] + 1][1], num=con_frames)]))
                self.angle[indices[0] - 1: indices[1] + 1] = np.squeeze(np.transpose(
                    np.array([np.linspace(self.angle[indices[0] - 1], self.angle[indices[1] + 1], num=con_frames)])))

    def divide_into_parts(self) -> list:
        """
        In order to treat the connections different than the actually tracked part, this function will split a single
        trajectory object into multiple trajectory objects.
        :return:
        """
        frame_dividers = [-1] + \
                         [i for i, (f1, f2) in enumerate(zip(self.frames, self.frames[1:])) if not f1 < f2] + \
                         [len(self.frames)]
        tracked_frames = [self.tracked_frames[i:i + 2] for i in range(0, len(self.tracked_frames), 2)]
        if len(self.VideoChain) == 0:
            self.VideoChain = [self.filename]
        parts = [Trajectory_part(self, [chain_element], [fr1 + 1, fr2 + 1], tracked_frame)
                 for chain_element, fr1, fr2, tracked_frame
                 in zip(self.VideoChain, frame_dividers, frame_dividers[1:], tracked_frames)]
        return parts

    def timer(self):
        """
        :return: time in seconds
        """
        return (len(self.frames) - 1) / self.fps

    def solving_time(self):
        return self.timer()

    def play_raw_movie(self):
        df = pd.read_excel(lists_exp_dir + '\\exp.xlsx')
        os.startfile(df[df['filename'] == self.filename]['directory'].iloc[0])

    def iterate_coords_for_ps(self, time_step: float = 1) -> iter:
        """
        Iterator over (x, y, theta) of the trajectory, time_step is given in seconds
        :return: tuple (x, y, theta) of the trajectory
        """
        number_of_frames = self.angle.shape[0]
        length_of_movie_in_seconds = number_of_frames/self.fps
        len_of_slicer = np.floor(length_of_movie_in_seconds/time_step).astype(int)

        slicer = np.cumsum([time_step*self.fps for _ in range(len_of_slicer)][:-1]).astype(int)
        for pos, angle in zip(self.position[slicer], self.angle[slicer]):
            yield pos[0], pos[1], angle

    def smoothed_pos_angle(self, position, angle, kernel_size):
        new_position = deepcopy(position)
        new_position[:, 0] = medfilt(position[:, 0], kernel_size=kernel_size)
        new_position[:, 1] = medfilt(position[:, 1], kernel_size=kernel_size)
        new_position[:, 0] = gaussian_filter(new_position[:, 0], sigma=kernel_size//5)
        new_position[:, 1] = gaussian_filter(new_position[:, 1], sigma=kernel_size//5)

        # unwrapped_angle = ConnectAngle(angle, self.shape)
        unwrapped_angle = np.unwrap(angle)
        new_unwrapped_angle = medfilt(unwrapped_angle, kernel_size=kernel_size)
        new_unwrapped_angle = gaussian_filter(new_unwrapped_angle, sigma=kernel_size//5)
        return new_position, new_unwrapped_angle

    def stuck(self, vel_norm=None, v_min=None) -> list:
        """
        :param v_min: minimal velocity in cm/s that will count as still moving.
        """

        if v_min is None:
            v_min = {'ant': 0.1, 'pheidole': 0.1, 'human': 0.1, 'humanhand': 0.5}[self.solver]

        slow_array = [v < v_min for v in np.abs(vel_norm)]
        attached_array = np.ones(len(slow_array)).astype(bool)
        stuck_array = np.logical_and(slow_array, attached_array)
        return stuck_array

    def has_forcemeter(self):
        return False

    def old_filenames(self, i: int):
        if i > 0:
            raise Exception('only one old filename available')
        return self.filename

    def speed(self):
        return np.linalg.norm(self.velocity(), axis=1)

    def velocity(self, fps=None, smoothed=2):
        av_rad = Maze(self).average_radius()
        if bool(smoothed):
            kernel_size = smoothed * self.fps
            # make kernel size odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            position, unwrapped_angle = self.smoothed_pos_angle(self.position, self.angle, int(kernel_size))
        else:
            position = self.position
            # unwrapped_angle = ConnectAngle(self.angle, self.shape)
            unwrapped_angle = np.unwrap(self.angle)
        args = (position[:, 0], position[:, 1], unwrapped_angle * av_rad)
        if fps is None:
            fps = self.fps

        return np.column_stack([np.diff(a) for a in args]) * fps

    def number_of_frames_in_part(self, i) -> int:
        """
        To find the length of a part of
        """
        lengths = [f1 for f1, f2 in zip(self.frames[:-1], self.frames[1:]) if f2 < f1]
        lengths.append(len(self.frames) - np.sum(lengths))
        return lengths[i]

    def play(self, wait: int = 0, cs=None, step=1, videowriter=False, frames=None, ts=None, geometry=None, bias=None):
        """
        Displays a given trajectory_inheritance (self)
        :param videowriter:
        :param frames:
        :param wait: how many milliseconds should we wait between displaying steps
        :param cs: Configuration space
        :param step: display only the ith frame
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

        if hasattr(x, 'participants') and x.participants is not None:
            x.participants.positions = x.participants.positions[f1:f2:step, :]
            x.participants.angles = x.participants.angles[f1:f2:step]
            if hasattr(x.participants, 'forces'):
                x.participants.forces.abs_values = x.participants.forces.abs_values[f1:f2:step, :]
                x.participants.forces.angles = x.participants.forces.angles[f1:f2:step, :]

        if geometry is None:
            geometry = x.geometry()

        my_maze = Maze(x, geometry=geometry)
        return x.run_trj(my_maze, display=Display(x.filename, x.fps, my_maze, wait=wait, cs=cs, videowriter=videowriter,
                                                  ts=ts))

    def check(self) -> None:
        """
        Simple check, whether the object makes sense. It would be better to create a setter function, that ensures, that
        all the attributes make sense...
        """
        if self.frames.shape != self.angle.shape:
            raise Exception('Your frame shape does not match your angle shape!')

    def cut_off(self, frame_indices=None, frames=None, time=-1):
        """

        :param frame_indices: frame indices (not the yellow numbers on top)
        :return:
        """
        new = copy(self)
        if frames is not None:
            raise Exception('frames is deprecated, use frame_indices instead') # I just did this to avoid mistakes
            ind_start = np.where(self.frames == frames[0])[0][time]
            ind_end = np.where(self.frames == frames[1])[0][time] + 1
            if ind_start > ind_end:
                ind_end = -1
                print('didnt reach the end')
            frame_indices = [ind_start, ind_end]
        new.frames = self.frames[frame_indices[0]:frame_indices[1]]
        new.position = self.position[frame_indices[0]:frame_indices[1]]
        new.angle = self.angle[frame_indices[0]:frame_indices[1]]
        if hasattr(self, 'ts') and self.ts is not None:
            new.ts = self.ts[frame_indices[0]:frame_indices[1]]
        if hasattr(self, 'OrderParam') and self.OrderParam is not None:
            new.OrderParam = self.OrderParam[frame_indices[0]:frame_indices[1]]
        return new

    def geometry(self):
        raise NotImplementedError
        # from DataFrame.import_excel_dfs import df_all, df_minimal
        # if len(df_all[df_all['filename'] == self.filename]) == 0:
        #     if self.solver == 'ant':
        #         return solver_geometry[self.solver]
        # return df_minimal[df_minimal['filename'] == self.filename].iloc[0][['maze dimensions', 'load dimensions']].tolist()

    def save(self, address=None) -> None:
        """
        1. save a pickle of the object
        2. save a pickle of a tuple of attributes of the object, in case I make a mistake one day, and change attributes
        in the class and then am incapable of unpickling my files.
        """
        self.check()
        dir = directory_results
        if self.solver == 'ant' and self.free:
            path.join(work_dir, 'Ant_Trajectories', 'Free')

        if address is None:
            address = dir + path.sep + self.filename

        # with open(address, 'wb') as f:
        #     try:
        #         self_copy = deepcopy(self)
        #         if hasattr(self_copy, 'participants'):
        #             delattr(self_copy, 'participants')
        #         pickle.dump(self_copy, f)
        #         print('Saving ' + self_copy.filename + ' in ' + address)
        #     except pickle.PicklingError as e:
        #         print(e)

        print('Saving minimal', self.filename, ' in path: ', dirs_exp_trajs[self.solver])
        pickle.dump((self.shape, self.size, self.solver, self.filename, self.fps,
                     self.position, self.angle, self.frames, self.winner),
                    open(address, 'wb'))

    def stretch(self, frame_number: int) -> None:
        """
        I have to interpolate a trajectory. I know the frame number and a few points, that the shape should walk
        through.
        I have to stretch the path to these points over the given number of frames.
        :param frame_number: number of frames the object is supposed to have in the end.
        """
        discont = np.pi / load_periodicity[self.shape]
        self.angle = np.unwrap(self.angle, discont=discont)
        stretch_factor = int(np.floor(frame_number / len(self.frames)))

        stretched_position = []
        stretched_angle = []
        if len(self.frames) == 1:
            stretched_position = np.vstack([self.position for _ in range(frame_number)])
            stretched_angle = np.vstack([self.angle for _ in range(frame_number)]).squeeze()

        for i, frame in enumerate(range(len(self.frames) - 1)):
            stretched_position += np.linspace(self.position[i], self.position[i + 1], stretch_factor,
                                              endpoint=False).tolist()
            stretched_angle += np.linspace(self.angle[i], self.angle[i + 1], stretch_factor, endpoint=False).tolist()

        self.position, self.angle = np.array(stretched_position), np.array(stretched_angle).squeeze()
        self.frames = np.array([i for i in range(self.angle.shape[0])])

    def load_participants(self):
        pass

    def averageCarrierNumber(self):
        pass

    def run_trj(self, my_maze, display=None):
        i = 0
        while i < len(self.frames) - 1:
            self.step(my_maze, i)
            i += 1
            if display is not None:
                end = display.update_screen(self, i)
                if end:
                    display.end_screen()
                    self.frames = self.frames[:i]
                    break
                display.renew_screen(movie_name=self.filename, frame_index=self.frames[display.i])
        if display is not None:
            display.end_screen()

    def initial_cond(self):
        """
        We changed the initial condition. First, we had the SPT start between the two slits.
        Later we made it start in the back of the room.
        :return: str 'back' or 'front' depending on where the shape started
        """
        if self.shape != 'SPT':
            return None
        elif self.position[0, 0] < Maze(self).slits[0]:
            return 'back'
        return 'front'

    def communication(self):
        return False

    def adapt_fps(self, new_fps: int) -> None:
        """
        param new_fps: if scale is larger than 1 it the fps will be reduced, if it is smaller than 1 it will be increased
        """
        # new_fps = self.fps / scale
        scale = self.fps / new_fps
        if scale > 1:
            scale = int(np.ceil(scale))
            self.fps = self.fps/scale  # this is supposed to be new_fps, frames per second
            self.position = self.position[::scale, :]  # np.array of x and y positions of the centroid of the shape
            self.angle = self.angle[::scale]  # np.array of angles while the shape is moving
            self.frames = self.frames[::scale]
            if self.participants is not None:
                raise ValueError('I have to change the participants as well')
        elif scale < 1:
            lengthen_by = int(1 / scale)
            # interpolate the position and angle
            if int(self.fps*lengthen_by) != new_fps:
                raise ValueError('I cannot interpolate the frames to the desired fps')
            self.fps = new_fps
            # calculate the new number of points
            new_size = len(self.position) * 3
            # create a new array with the interpolated points
            new_array = np.zeros((new_size, 2))
            new_array[:, 0] = np.interp(np.linspace(0, len(self.position) - 1, lengthen_by * len(self.position)),
                                        np.arange(len(self.position)), self.position[:, 0])
            new_array[:, 1] = np.interp(np.linspace(0, len(self.position) - 1, lengthen_by * len(self.position)),
                                        np.arange(len(self.position)), self.position[:, 1])
            self.position = new_array
            self.angle = np.interp(np.linspace(0, len(self.angle) - 1, lengthen_by * len(self.angle)),
                                   np.arange(len(self.angle)), self.angle)
            self.frames = np.interp(np.linspace(0, len(self.frames) - 1, lengthen_by * len(self.frames)),
                                    np.arange(len(self.frames)), self.frames).astype(int)
            DEBUG = 1


class Trajectory_part(Trajectory):
    def __init__(self, parent_traj, VideoChain: list, indices: iter, tracked_frames: list, parent_states=None):
        """

        :param parent_traj: trajectory that the part is taken from
        :param VideoChain: list of names of videos that are supposed to be part of the trajectory part
        :param indices: []
        :param tracked_frames: list of frames that are tracked
        :param parent_states: list of states of parents
        :param parent_vel: velocity of the parent
        """
        super().__init__(size=parent_traj.size, shape=parent_traj.shape, solver=parent_traj.solver,
                         filename=parent_traj.filename, fps=parent_traj.fps, winner=parent_traj.winner,
                         VideoChain=VideoChain)
        if len(indices) == 2 and indices[0] != indices[1]-1:
            raise ValueError('The indices are not consecutive')
        indices = [int(i) for i in indices]
        self.parent_traj = parent_traj
        self.frames_of_parent = indices
        self.frames = parent_traj.frames[indices]
        self.position = parent_traj.position[indices]
        self.angle = parent_traj.angle[indices]
        self.tracked_frames = tracked_frames
        self.falseTracking = []
        if parent_states is not None:
            self.states = [parent_states[i] for i in indices]
        else:
            self.states = None

    def is_connector(self):
        if self.VideoChain[-1] is None:
            return False
        return 'CONNECTOR' in self.VideoChain[-1]

    def geometry(self):
        return self.parent_traj.geometry()

    def __add__(self, file2):
        file12 = deepcopy(self)
        file12.position = np.vstack((self.position, file2.position))

        per = 2 * np.pi / load_periodicity[file12.shape]
        a0 = np.floor(self.angle[-1] / per) * per + np.mod(file2.angle[1], per)
        file12.angle = np.hstack((self.angle, file2.angle - file2.angle[1] + a0))
        file12.frames = np.hstack((self.frames, file2.frames))
        file12.tracked_frames = file12.tracked_frames + file2.tracked_frames

        # if not self.free:
        #     # file12.contact = self.contact + file2.contact  # We are combining two lists here...
        #     # file12.state = np.hstack((np.squeeze(self.state), np.squeeze(file2.state)))
        #     file12.winner = file2.winner  # The success of the attempt is determined, by the fact that the last file
        #     # is either winner or looser.

        file12.VideoChain = self.VideoChain + file2.VideoChain
        file12.falseTracking = self.falseTracking + file2.falseTracking

        if self.states is not None:
            file12.states = self.states + file2.states
        else:
            file12.states = None

        # Delete the load of filename
        return file12

    def split_at_directional_change_in_turning(self) -> list:
        """
        Split the trajectory into continuous movements
        :param axis: axis along which the movement is continuous
        :return: list of Trajectory_parts
        """
        traj_parts = []
        traj_copy = deepcopy(self)
        # traj_copy.smooth(sec_smooth=1)
        vel = np.diff(traj_copy.angle)
        # find all the indices where the absolute velocity is below 0.005
        if self.solver == 'human':
            standing_still = np.where([np.abs(vel) < 1/1000 * 10])[1]
        if self.solver == 'ant':
            standing_still = np.where([np.abs(vel) < 1 / 1000 * 10])[1]
        # group the indices into continuous blocks
        standing_still_episodes = np.split(standing_still, np.where(np.diff(standing_still) != 1)[0] + 1)

        # for every standing still episode, split in half and set the velocity to 0.1 or -0.1
        # depending on the sign of the velocity

        standing_still_episodes = [sse for sse in standing_still_episodes if len(sse) > 1]

        if len(standing_still_episodes) == 0:
            return [traj_copy]

        # completely standing still
        if len(standing_still_episodes[0]) == len(vel):
            return []

        if standing_still_episodes[0][0] == 0:
            vel[standing_still_episodes[0]] = np.sign(vel[standing_still_episodes[0][-1]+1]) * 0.000001

        if standing_still_episodes[-1][-1] == len(vel) - 1:
            vel[standing_still_episodes[-1]] = np.sign(vel[standing_still_episodes[-1][0]-1]) * 0.000001

        for episode in standing_still_episodes:
            if len(episode) > 1:
                half_mark = episode[0] + int(len(episode)/2)
                vel[episode[0]: half_mark] = np.sign(vel[episode[0]]) * 0.000001
                vel[half_mark: episode[-1]] = np.sign(vel[episode[-1]]) * 0.000001

        # find the indices where the vel changes sign and goes beyond absolute value of 0.1
        direction_change_indices = np.where(np.abs(np.diff(np.sign(vel))) > 1)[0]
        direction_change_indices = np.hstack((0, direction_change_indices, len(vel) - 1))

        # plt.close()
        # plt.plot(self.frames[:-1], vel * 1000, label='vel * 1000, fixed')
        # plt.plot(self.frames[:-1], np.diff(traj_copy.angle) * 1000, label='vel * 1000')
        # plt.plot(self.frames, traj_copy.angle, label='angle')
        # plt.plot(self.frames[direction_change_indices], vel[direction_change_indices] * 100, 'ro')
        # # mark the frames where standing_still with red background
        # for episode in standing_still_episodes:
        #     plt.axvspan(self.frames[episode[0]], self.frames[episode[-1]], facecolor='r', alpha=0.1)
        # plt.legend()
        # # set y limit to -pi to pi
        # plt.ylim(-6 * np.pi, 6 * np.pi)

        if len(direction_change_indices) > 2:
            print('There are more than two direction changes in the trajectory')

        for (start, end) in zip(direction_change_indices[:-1], direction_change_indices[1:]):
            traj_parts.append(Trajectory_part(self, indices=range(start, end), VideoChain=[], tracked_frames=[],
                                              parent_states=self.states))
        return traj_parts