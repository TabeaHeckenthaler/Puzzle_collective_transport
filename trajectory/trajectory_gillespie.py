from trajectory.trajectory_general import Trajectory
import numpy as np
from Setup.Maze import Maze
from PhysicsEngine.drawables import Arrow, Point

time_step = 0.01


class Trajectory_gillespie(Trajectory):
    def __init__(self, shape=None, size=None, position=None, angle=None, frames=None, winner=None,
                 filename='gillespie_test', fps=time_step, free=False, nAtt=None, nPullers=None, Ftot=None,
                 OrderParam=None
                 ):

        super().__init__(size=size, shape=shape, solver='gillespie', filename=filename, fps=int(fps), winner=winner,
                         position=position, angle=angle, frames=frames)

        self.free = free
        self.gillespie = None
        self.nAtt = nAtt
        self.nPullers = nPullers
        self.Ftot = Ftot
        self.OrderParam = OrderParam

    def step_simulation(self, my_maze, i, display=None):

        my_maze.set_configuration(self.position[i], self.angle[i])

        if self.gillespie.time_until_next_event < time_step:
            self.gillespie.time_until_next_event = self.gillespie.whatsNext(my_maze.bodies[-1])

        self.forces(my_maze.bodies[-1], display=display)

        self.gillespie.time_until_next_event -= time_step
        my_maze.Step(time_step, 10, 10)

        self.position = np.vstack((self.position, [my_maze.bodies[-1].position.x, my_maze.bodies[-1].position.y]))
        self.angle = np.hstack((self.angle, my_maze.bodies[-1].angle))
        return

    def forces(self, my_load, pause=False, display=None):
        # TODO: make this better...
        my_load.linearVelocity = 0 * my_load.linearVelocity
        my_load.angularVelocity = 0 * my_load.angularVelocity

        """ Magnitude of forces """

        for i in range(len(self.gillespie.n_p)):
            start = self.gillespie.attachment_site_world_coord(my_load, i)
            end = None

            if self.gillespie.n_p[i]:
                f_x, f_y = self.gillespie.ant_force(my_load, i, pause=pause)
                Arrow(start, start + [100 * f_x, 100 * f_y], 'puller').draw(display)

            elif self.gillespie.n_l[i]:
                Point(start, end).draw(display)

            else:
                Point(start, end).draw(display)

    def run_simulation(self, frameNumber):
        my_maze = Maze(size=self.size, shape=self.shape, solver='sim')
        self.frames = np.linspace(1, frameNumber, frameNumber)
        self.position = np.array([[my_maze.arena_length / 4, my_maze.arena_height / 2]])
        self.angle = np.array([0], dtype=float)  # array to store the position and angle of the load
        from PhysicsEngine.Display import Display
        i = 0
        display = Display(self.filename, self.fps, my_maze, wait=10)
        while i < len(self.frames) - 1:
            self.step_simulation(my_maze, i, display=display)
            i += 1
            if display is not None:
                end = display.update_screen(self, i)
                if end:
                    display.end_screen()
                    self.frames = self.frames[:i]
                    break
                display.renew_screen(movie_name=self.filename)
        if display is not None:
            display.end_screen()

    def load_participants(self):
        self.participants = self.gillespie

    def averageCarrierNumber(self):
        return np.mean(self.nAtt)

    def geometry(self) -> tuple:
        return ('MazeDimensions_new2021_SPT_ant_Amir.xlsx',
                'LoadDimensions_new2021_SPT_ant_Amir.xlsx')

    def iterate_coords_for_ps(self, time_step: float = 1) -> iter:
        """
        Iterator over (x, y, theta) of the trajectory, time_step is given in seconds
        :return: tuple (x, y, theta) of the trajectory
        """
        number_of_frames = self.angle.shape[0]
        length_of_movie_in_seconds = number_of_frames/self.fps
        len_of_slicer = np.floor(length_of_movie_in_seconds/time_step).astype(int)

        # because we use the Medium sized phase space
        # position = self.position * {'XL': 1/4, 'L': 1/2, 'M': 1, 'S': 2, 'XS': 4}[self.size]

        slicer = np.cumsum([time_step*self.fps for _ in range(len_of_slicer)][:-1]).astype(int)
        for pos, angle in zip(self.position[slicer], self.angle[slicer]):
            yield pos[0], pos[1], angle
