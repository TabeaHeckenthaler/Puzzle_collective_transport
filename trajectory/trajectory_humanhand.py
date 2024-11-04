from trajectory.trajectory_general import Trajectory
from os import path
import pickle
import numpy as np
from openpyxl import load_workbook
import pandas as pd

trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'
length_unit = 'cm'
excel_dir = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Hand Experiments'.format(path.sep, path.sep, path.sep,
                                                                                path.sep, path.sep)


class Trajectory_humanhand(Trajectory):
    def __init__(self, size=None, shape=None, filename=None, fps=50, winner=bool, x_error=None, y_error=None,
                 angle_error=None, falseTracking=None):

        super().__init__(size=size, shape=shape, solver='humanhand', filename=filename, fps=fps, winner=winner)
        self.x_error = x_error
        self.y_error = y_error
        self.angle_error = angle_error
        self.falseTracking = falseTracking
        self.tracked_frames = []
        self.state = np.empty((1, 1), int)

    def matlabFolder(self):
        return trackedHumanHandMovieDirectory

    def matlab_loading(self, old_filename):
        load_center, angle, frames, videoChain = pickle.load(open(self.matlabFolder() + path.sep + self.filename + '.pkl', 'rb'))

        if len(load_center) == 2:
            self.position = np.array([load_center])
            self.angle = np.array([angle])
        else:
            self.position = np.array(load_center)  # array to store the position and angle of the load
            self.angle = np.array(angle)
        self.frames = np.array(frames)
        self.VideoChain = videoChain
        # self.interpolate_over_NaN()

    def load_participants(self):
        self.participants = Humanhand(self)

    def averageCarrierNumber(self):
        return 1

    def geometry(self) -> tuple:
        return 'MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx'

    def add_missing_frames(self, chain):
        pass


class Humanhand:
    def __init__(self, filename):
        self.filename = filename
        return


class ExcelSheet:
    def __init__(self):
        self.sheet = pd.DataFrame(load_workbook(filename=excel_dir + path.sep + "video_data.xlsx").active.values)
        self.sheet.columns = self.sheet.iloc[0]
        self.sheet = self.sheet[1:]

    def get_experiments(self):
        for i in range(1, len(self.sheet['raw video name'])):
            movies = self.sheet['raw video name'][i].value.split('; ')
            frames_string = self.sheet['frames'][i].value.split('; ')
            frames = [list(range(*list(map(int, frame.split(', '))))) for frame in frames_string]
            light = self.sheet['light'][i].value != 'n'
            movies_frames = dict(zip(movies, frames))
            eyesight = self.sheet['eyesight'][i].value == 'y'

            if self.sheet['frames to exclude'][i].value:
                movie = self.sheet['frames to exclude'][i].value.split(': ')[0]
                frames_to_remove = list(range(*[int(f) for f in self.sheet['F'][i].value.split(': ')[1].split(', ')]))
                movies_frames[movie] = [fr for fr in movies_frames[movie] if fr not in frames_to_remove]

            filename = movies[0] + '_' + str(frames[0][0])

    def with_eyesight(self, filename) -> bool:
        [first_movie, first_frame] = filename.split('_')
        k = self.sheet[(self.sheet['raw video name'].apply(lambda x: x.split('; ')[0]) == first_movie) &
                        (self.sheet['frames'].apply(lambda x: x.split(', ')[0]) == first_frame)]
        return k['eyesight'].iloc[0] == 'y'
    
        
if __name__ == '__main__':
    e = ExcelSheet()
    print(e.with_eyesight('YI029701_6249'))
