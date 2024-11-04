import numpy as np
from general_functions import flatten
from trajectory.get import get
import pickle
from os import path
from matplotlib import pyplot as plt
from skimage import measure
from PIL import Image, ImageDraw
import json
from directories import ConfSpace_Directory, home
from Setup.Load import loops
from Setup.Maze import Maze
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


with open(path.join(home, 'Setup', 'ResizeFactors.json'), "r") as read_content:
    ResizeFactors = json.load(read_content)


def resolution(geometry: tuple, size: str, solver: str, shape: str):
    if geometry == ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'):
        Res = {'Large': 0.1, 'L': 0.1, 'Medium': 0.07, 'M': 0.07, 'Small Far': 0.02, 'Small Near': 0.02, 'S': 0.02}
        return Res[size] * 0.5  # had to add twice the resolution, because the fast marching method was not able to pass

    # print('My resolution is to high')
    if shape == 'SPT':
        return 0.1 * ResizeFactors[solver][size]  # used to be 0.1
    return 0.1 * ResizeFactors[solver][size]


class ConfigSpace(object):
    def __init__(self, space: np.array, name=''):
        self.space = space  # True, if configuration is possible; False, if there is a collision with the wall
        self.name = name
        self.dual_space = None


class ConfigSpace_Maze(ConfigSpace):
    def __init__(self, solver: str, size: str, shape: str, geometry: tuple, name="", x_range=None, y_range=None,
                 pos_resolution=None, theta_resolution=None, space=None, maze=None):
        """

        :param solver: type of solver (ps_simluation, ant, human, etc.)
        :param size: size of the maze (XL, L, M, S)
        :param shape: shape of the load in the maze (SPT, T, H ...)
        :param geometry: tuple with names of the .xlsx files that contain the relevant dimensions
        :param name: name of the PhaseSpace.
        """
        super().__init__(space)

        if len(name) == 0:
            name = size + '_' + shape

        self.name = name
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry = geometry

        if maze is None:
            maze = Maze(size=size, shape=shape, solver=solver, geometry=geometry)
        if geometry == ('MazeDimensions_new2021_SPT_ant_perfect_scaling.xlsx',
                        'LoadDimensions_new2021_SPT_ant_perfect_scaling.xlsx'):
            factor = {'XL': 4, 'L': 2, 'M': 1, 'S': 0.5, 'XS': 0.25}[size]
            maze_M = Maze(size='M', shape=shape, solver=solver, geometry=geometry)
            if x_range is None:
                x_range = (0, (maze_M.slits[-1] + max(maze_M.getLoadDim()) + 1)*factor)
            if y_range is None:
                y_range = (0, maze_M.arena_height * factor)
        else:
            if x_range is None:
                x_range = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
            if y_range is None:
                y_range = (0, maze.arena_height)

        self.extent = {'x': x_range,
                       'y': y_range,
                       'theta': (0, np.pi * 2)}
        self.average_radius = maze.average_radius()

        if pos_resolution is None:
            pos_resolution = self.extent['y'][1] / self.number_of_points()['y']
        self.pos_resolution = pos_resolution
        if theta_resolution is None:
            theta_resolution = 2 * np.pi / self.number_of_points()['theta']
        self.theta_resolution = theta_resolution

        self.space_boundary = None
        self.fig = None

        load = maze.bodies[-1]
        maze_corners = np.array_split(maze.corners(), maze.corners().shape[0] // 4)
        load_corners = np.array(flatten(loops(load)))
        # loop_indices = [0, 1, 2, 3, 0]

        rect_edge_indices = np.array(((0, 1), (1, 2), (2, 3), (3, 0)))

        self.load_points = []
        self.load_edges = []
        for i, load_vertices_list in enumerate(np.array_split(load_corners, int(load_corners.shape[0] / 4))):
            self.load_points.extend(load_vertices_list)
            self.load_edges.extend(rect_edge_indices + 4 * i)
        self.load_points = np.array(self.load_points, float)

        self.maze_points = []
        self.maze_edges = []
        for i, maze_vertices_list in enumerate(maze_corners):
            self.maze_points.extend(maze_vertices_list)
            self.maze_edges.extend(rect_edge_indices + 4 * i)
        self.eroded_space = None

    def directory(self) -> str:
        if self.size in ['Small Far', 'Small Near']:  # both have same dimensions
            filename = 'Small' + '_' + self.shape + '_' + self.geometry[0][:-5]
        else:
            filename = self.size + '_' + self.shape + '_' + self.geometry[0][:-5]

        path_ = path.join(ConfSpace_Directory, filename + '.pkl')
        return path_

    def get_verts_faces_of_states_directory(self, dilation_radius=1, boundary=True, transition=False):
        if boundary:
            string = 'boundary_verts_faces'
            assert not transition
        elif transition:
            string = 'transition_verts_faces'
            assert not boundary
        else:
            string = 'space_verts_faces'

        if dilation_radius > 1:
            string += '_dilated_' + str(dilation_radius)

        filename = '_'.join([self.size, self.solver, self.geometry[0][:-5], string]) + '.pkl'
        directory = '\\'.join(self.directory().split('\\')[:-1] + [string, filename])
        return directory

    def initialize_maze_edges(self) -> None:
        """
        set x&y edges to 0 in order to define the maze boundaries (helps the visualization)
        :return:
        """
        self.space[0, :, :] = False
        self.space[-1, :, :] = False
        self.space[:, 0, :] = False
        self.space[:, -1, :] = False

    def load_space(self,directory=None) -> None:
        """
        Load Phase Space pickle.
        """
        if directory is not None:
            pass
        else:
            directory = self.directory()
            # after the last '\\' add \
        if path.exists(directory):
            (self.space, self.space_boundary, self.extent) = pickle.load(open(directory, 'rb'))
            self.initialize_maze_edges()
            if self.extent['theta'] != (0, 2 * np.pi):
                print('need to correct' + self.name)
        else:
            raise ValueError('No such file: ' + directory + ' run calculate_space() and save_space() first')
        return

    def get_space_verts_faces(self, boundary=False, dilation_itter=1, transition=False, step_size=1):
        directory = self.get_verts_faces_of_states_directory(dilation_radius=dilation_itter, boundary=boundary,
                                                             transition=transition)
        space = self.space
        if not path.exists(directory):
            self.save_verts_faces(space, directory, step_size=step_size)
        with open(directory, 'rb') as f:
            space, verts, faces = pickle.load(f)
        return space, verts, faces

    def save_verts_faces(self, space, directory, step_size=1):
        verts, faces, _, _ = measure.marching_cubes(space, level=0, step_size=step_size, allow_degenerate=True)
        # scale the vertices[0, :] to be in the range of the maze
        verts[:, 0] = verts[:, 0] * self.extent['x'][1] / space.shape[0]
        verts[:, 1] = verts[:, 1] * self.extent['y'][1] / space.shape[1]
        verts[:, 2] = verts[:, 2] * self.extent['theta'][1] * self.average_radius / space.shape[2]

        assert directory[-4:] == '.pkl', 'directory should end with .pkl'
        # save verts and faces in pkl
        with open(directory, 'wb') as f:
            pickle.dump((space, verts, faces), f)
            f.close()

    def number_of_points(self) -> dict:
        """
        How to pixelize the PhaseSpace. How many pixels along every axis.
        :return: dictionary with integers for every axis.
        """
        # x_num = np.ceil(self.extent['x'][1]/resolution)
        res = resolution(self.geometry, self.size, self.solver, self.shape)
        y_num = np.ceil(self.extent['y'][1] / res)
        theta_num = np.ceil(self.extent['theta'][1] * self.average_radius / res)
        return {'x': None, 'y': y_num, 'theta': theta_num}

    def empty_space(self) -> np.array:
        return np.zeros((int(np.ceil((self.extent['x'][1] - self.extent['x'][0]) / float(self.pos_resolution))),
                         int(np.ceil((self.extent['y'][1] - self.extent['y'][0]) / float(self.pos_resolution))),
                         int(np.ceil(
                             (self.extent['theta'][1] - self.extent['theta'][0]) / float(self.theta_resolution)))),
                        dtype=bool)

    def calc_theta_slice(self, theta, res_x, res_y, xbounds, ybounds):
        arr = np.ones((res_x, res_y), bool)
        im = Image.fromarray(arr)  # .astype('uint8')?
        draw = ImageDraw.Draw(im)

        s, c = np.sin(theta), np.cos(theta)
        rotation_mat = np.array(((c, -s), (s, c)))
        load_points = (rotation_mat @ self.load_points.T).T

        for maze_edge in self.maze_edges:
            maze_edge = (self.maze_points[maze_edge[0]],
                         self.maze_points[maze_edge[1]])
            for load_edge in self.load_edges:
                load_edge = (load_points[load_edge[0]],
                             load_points[load_edge[1]])
                self.imprint_boundary(draw, arr.shape, load_edge, maze_edge, xbounds, ybounds)

        return np.array(im)  # type: ignore  # this is the canonical way to convert Image to ndarray

    @staticmethod
    def imprint_boundary(draw, shape, edge_1, edge_2, xbounds, ybounds):
        """
        Takes arr, and sets to 0 all pixels which intersect/lie inside the quad roughly describing
        the pixels which contain a point such that a shift by it causes the two edges to intersect

        @param draw: PIL ImageDraw object
        @param shape: the image shape (res_y, res_x)
        @param edge_1: first edge
        @param edge_2: second edge
        """

        # Reflected Binary Code~
        points = tuple(p + edge_2[0] for p in edge_1) + tuple(p + edge_2[1] for p in edge_1[::-1])

        # project into array space
        points = np.array(points)
        points[:, 0] -= xbounds[0]
        points[:, 0] *= shape[0] / (xbounds[1] - xbounds[0])
        points[:, 1] -= ybounds[0]
        points[:, 1] *= shape[1] / (ybounds[1] - ybounds[0])
        points += .5
        points = points.astype(int)  # round to nearest integer

        draw.polygon(tuple(points[:, ::-1].flatten()), fill=0, outline=0)

    @staticmethod
    def shift_by_pi(space):
        middle = space.shape[2] // 2
        space = np.concatenate([space[:, :, middle:], space[:, :, :middle], ], axis=-1)
        return space

    def calculate_space(self) -> None:
        """
        This module calculated a space.
        param point_particle:
        :return:
        This was beautifully created by Rotem!!!
        """
        maze = Maze(size=self.size, shape=self.shape, solver=self.solver, geometry=self.geometry)
        # use same shape and bounds as original phase space calculator
        # space_shape = (415, 252, 616)  # x, y, theta.
        space_shape = self.empty_space().shape  # x, y, theta.

        xbounds = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
        ybounds = (0, maze.arena_height)

        final_arr = np.empty(space_shape, bool)  # better to use space_shape[::-1] in terms of access speed
        thet_arr = np.linspace(0, 2 * np.pi, space_shape[2], False)

        # make the final array slice by slice
        for i, theta in enumerate(thet_arr):
            if not i % 50:
                print(f"{i}/{space_shape[2]}")
            final_arr[:, :, i] = self.calc_theta_slice(theta, space_shape[0], space_shape[1], xbounds, ybounds)
        self.space = self.shift_by_pi(final_arr)  # somehow this was shifted by pi...

    def calculate_boundary(self) -> None:
        if self.space is None:
            self.calculate_space()
        self.space_boundary = self.empty_space()

    def save_space(self, directory: str = None) -> None:
        """
        Pickle the numpy array in given path, or in default path. If default directory exists, add a string for time, in
        order not to overwrite the old .pkl file.
        :param directory: Where you would like to save.
        """
        if not hasattr(self, 'space') and self.space is not None:
            self.calculate_space()
        if not hasattr(self, 'space_boundary') and self.space_boundary is not None:
            self.calculate_boundary()
        if directory is None:
            if path.exists(self.directory()):
                directory = self.directory()
            else:
                directory = self.directory()
        print('Saving ' + self.name + ' in path: ' + directory)
        pickle.dump((np.array(self.space, dtype=bool),
                     np.array(self.space_boundary, dtype=bool),
                     self.extent),
                    open(directory, 'wb'))

    @staticmethod
    def plot_mesh(fig, vertices, faces, colors, ax=None, labels=True):
        """

        :param ax: axis to plot on
        :param vertices: First element in vs determines the scale, vertices
        :param faces: faces
        :param colors: colors
        :return:
        """
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        for v, f, fc in zip(vertices, faces, colors):
            mesh = Poly3DCollection(v[f], linewidths=0.01, edgecolors='k', cmap='viridis')
            mesh.set_facecolor(fc)  # Set facecolor with alpha for transparency
            ax.add_collection3d(mesh)

        # change range so that the plot is centered
        ax.grid(False)
        scale = vertices[0].flatten()
        lims = [[scale[i::3].min(), scale[i::3].max()] for i in range(3)]
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])  # (2.281670423183923, 1.4471229866773272, 2.2161840150854024)
        ax.set_box_aspect((lims[0][1] - lims[0][0], lims[1][1] - lims[1][0], lims[2][1] - lims[2][0]))

        if labels:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('theta * average radius')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    def plot(self, fig, ax):
        _, verts, faces = self.get_space_verts_faces(boundary=False, transition=False)

        colors = [(0.7, 0.7, 0.7, 0.02)]
        self.plot_mesh(fig, [verts], [faces], colors, ax=ax, labels=False)

        # no grey background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # turn off the axes
        ax.set_axis_off()

        ax.view_init(elev=0, azim=-90)
        ax.view_init(elev=7.948051948051955, azim=-54.467532467532465)

    def plot_traj_in_fig(self, coords, ax):
        # reduce coords to maximally 1000 points
        if coords.shape[0] > 2000:
            coords = coords[::int(np.ceil(coords.shape[0] / 2000))]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2] * self.average_radius,
                   marker='o', color='red', alpha=0.7, s=3)


if __name__ == '__main__':
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

    cs = ConfigSpace_Maze('ant', 'XL', 'H', ('MazeDimensions_ant.xlsx', 'LoadDimensions_ant.xlsx'))
    cs.calculate_space()
    cs.save_space()
    cs.load_space()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    cs.plot(fig, ax)
    plt.savefig(home + '\\ConfigSpace\\Figures\\CS_H_XL.png', dpi=300, transparent=True)
    plt.close()


