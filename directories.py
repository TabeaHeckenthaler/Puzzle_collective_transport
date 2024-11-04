from os import path, mkdir

# home = 'C:\\Users\\tabea\\PycharmProjects\\Puzzle_collective_transport\\'
home = path.join(path.abspath(__file__).split('\\')[0] + path.sep, *path.abspath(__file__).split(path.sep)[1:-1])
# data_home = '\\\\phys-guru-cs\\ants\\Tabea\\PyCharm_Data\\AntsShapes\\'
data_home = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'PyCharm_Data', 'AntsShapes')

network_dir = path.join(data_home, 'Time_Series')
# ConfSpace_Directory = path.join(data_home, 'Configuration_Spaces')
ConfSpace_Directory = path.join(home, 'ConfigSpace', 'results')
work_dir = path.join(data_home, 'Pickled_Trajectories')
dir_gillespie_trajs = path.join(work_dir, 'Gillespie_Trajectories')
mini_work_dir = path.join(data_home, 'mini_Pickled_Trajectories')
dirs_exp_trajs = {'ant': path.join(mini_work_dir, 'Ant_Trajectories'),
                  'pheidole': path.join(mini_work_dir, 'Pheidole_Trajectories'),
                  'human': path.join(mini_work_dir, 'Human_Trajectories'),
                  'humanhand': path.join(mini_work_dir, 'HumanHand_Trajectories'),
                  'gillespie': path.join(mini_work_dir, 'Gillespie_Trajectories')}
df_dir = path.join(home, 'lists_of_experiments', 'data_frame.xlsx')
video_directory = path.join(home, 'Videos')
maze_dimension_directory = path.join(home, 'Setup')
excel_sheet_directory = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'Human Experiments')
lists_exp_dir = path.join(data_home, 'DataFrame', 'excel_experiment_lists')


original_movies_dir_ant = [
    '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Videos'.format(path.sep, path.sep, path.sep, path.sep, path.sep),
    '{0}{1}phys-guru-cs{2}ants{3}Lena{4}Movies'.format(path.sep, path.sep, path.sep, path.sep, path.sep),
    '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes'.format(path.sep, path.sep, path.sep, path.sep, path.sep)]
original_movies_dir_human = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Raw Data and Videos'.format(
    path.sep, path.sep, path.sep, path.sep, path.sep, path.sep)
original_movies_dir_humanhand = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Hand Experiments{5}Raw Data{6}' \
                                '2022_04_04 (Department Retreat)'.format(path.sep, path.sep, path.sep, path.sep,
                                                                         path.sep, path.sep, path.sep)

trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Alumni_Students{4}Aviram{5}Shapes Results'.format(path.sep,
                                                                                                          path.sep,
                                                                                                          path.sep,
                                                                                                          path.sep,
                                                                                                          path.sep,
                                                                                                          path.sep)
trackedPheidoleMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Pheidole Shapes Results'.format(path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep,
                                                                                                      path.sep)

trackedHumanMovieDirectory = path.join(excel_sheet_directory, 'Output')
trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'  # TODO


def MatlabFolder(solver, size, shape, free=False):
    if solver == 'ant':
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        if not free:
            return path.join(trackedAntMovieDirectory, 'Slitted', shape_folder_naming[shape], size, 'Output Data')
        else:
            return path.join(trackedAntMovieDirectory, 'Free', 'Output Data', shape_folder_naming[shape])

    if solver == 'pheidole':
        return path.join(trackedPheidoleMovieDirectory, size, 'Output Data')

    if solver == 'human':
        return path.join(trackedHumanMovieDirectory, size, 'Data')

    if solver == 'humanhand':
        return trackedHumanHandMovieDirectory

    else:
        print('MatlabFolder: who is solver?')


def NewFileName(old_filename: str, solver: str, size: str, shape: str, expORsim: str) -> str:
    import glob
    if expORsim == 'sim':
        counter = int(len(glob.glob(size + '_' + shape + '*_' + expORsim + '_*')) / 2 + 1)
        return size + '_' + shape + '_sim_' + str(counter)
    if expORsim == 'exp':
        filename = old_filename.replace('.mat', '')
        if shape.endswith('ASH'):
            return filename.replace(old_filename.split('_')[0], size + '_' + shape)

        else:
            if solver in ['ant', 'pheidole']:
                if size + shape in filename or size + '_' + shape in filename:
                    return filename.replace(size + shape, size + '_' + shape)
                else:
                    raise ValueError('Your filename does not seem to be right.')
            elif solver in ['human', 'humanhand']:
                return filename
