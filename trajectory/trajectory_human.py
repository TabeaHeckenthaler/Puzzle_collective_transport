import os
from trajectory.trajectory_general import Trajectory
import numpy as np
from os import path, listdir
import scipy.io as sio
from trajectory.humans import Humans
from directories import MatlabFolder
from trajectory.forces import get_sheet
from trajectory.humans import get_excel_worksheet_index

length_unit = 'm'
# these are perfect solutions of the puzzles
perfect_filenames = ['large_20210805171741_20210805172610_perfect',
                     'medium_20210507225832_20210507230303_perfect',
                     'small2_20220308120548_20220308120613_perfect']


class Trajectory_human(Trajectory):
    def __init__(self, size=None, shape=None, filename=None, fps=30, winner=bool, x_error: float = 0,
                 y_error: float = 0, angle_error: float = 0, falseTracking: list = [], VideoChain=str(), position=None,
                 angle=None, frames=None, tracked_frames=None):

        super().__init__(size=size, shape=shape, solver='human', filename=filename, fps=fps, winner=winner,
                         position=position, angle=angle, frames=frames)

        self.x_error = x_error
        self.y_error = y_error
        self.angle_error = angle_error
        self.falseTracking = falseTracking
        self.tracked_frames = tracked_frames
        self.state = np.empty((1, 1), int)
        self.VideoChain = VideoChain  # this is an evil artifact. I don't want to have this attribute
        self.communication = self.communication()
        self.forcemeter = self.has_forcemeter()  # TODO: this is always false currently...

    def geometry(self) -> tuple:
        return 'MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'

    def has_forcemeter(self) -> bool:
        """
        Checks whether experiment has force meters
        """
        sheet = get_sheet()
        return sheet.cell(row=get_excel_worksheet_index(self.filename), column=19).value != '/'

    def matlab_loading(self, old_filename):
        folder = MatlabFolder(self.solver, self.size, self.shape)

        if old_filename + '.mat' in listdir(folder):
            file = sio.loadmat(folder + path.sep + old_filename)
        else:
            raise Exception('Cannot find ' + old_filename + '.mat' + ' in ' + str(folder))

        load_center = file['load_CoM'][:, 2:4]
        load_center[:, 0] = load_center[:, 0] + self.x_error
        load_center[:, 1] = load_center[:, 1] + self.y_error
        shape_orientation = np.matrix.transpose(file['orientation'][:] * np.pi / 180 + self.angle_error)[0]

        self.frames = np.linspace(1, load_center.shape[0], load_center.shape[0]).astype(int)

        if load_center.size == 2:
            self.position = np.array([load_center])
            self.angle = np.array([shape_orientation])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(shape_orientation)

        for frames in self.falseTracking:
            self.position[frames[0]: frames[1]] = np.linspace(self.position[frames[0]], self.position[frames[1]],
                                                              num=frames[1]-frames[0])
            self.angle[frames[0]: frames[1]] = np.linspace(self.angle[frames[0]], self.angle[frames[1]],
                                                           num=frames[1]-frames[0])

        self.interpolate_over_NaN()

    def open_tracked_video(self):
        directory1 = '\\\\phys-guru-cs\\ants\\Tabea\\Human Experiments\\Output\\' + self.size + \
                     '\\Videos\\' + self.filename + '_Corrected.avi'
        directory2 = '\\\\phys-guru-cs\\ants\\Tabea\\Human Experiments\\Output\\' + self.size + \
                     '\\Videos\\' + self.filename + '_Full.avi'

        if path.exists(directory1):
            os.startfile(directory1)
        elif path.exists(directory2):
            os.startfile(directory2)
        else:
            raise Exception('Cannot find video ' + directory1)

    def communication(self):
        index = get_excel_worksheet_index(self.filename)
        return bool(get_sheet().cell(row=index, column=5).value == 'C')

    def load_participants(self):
        if not hasattr(self, 'participants') or self.participants is None:
            self.participants = Humans(self)

    def averageCarrierNumber(self):
        if not hasattr(self, 'participants'):
            self.load_participants()
        return len(self.participants.occupied)
