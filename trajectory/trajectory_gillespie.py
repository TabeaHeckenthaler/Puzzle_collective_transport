from trajectory.trajectory_general import Trajectory
import numpy as np
from Setup.Maze import Maze
from PhysicsEngine.drawables import Arrow, Point
from Box2D import b2Vec2
from PhysicsEngine.Display import Display
from Setup.Load import init_sites
from scipy.stats import vonmises


class Trajectory_gillespie(Trajectory):
    def __init__(self, shape=None, size=None, position=None, angle=None, frames=None, winner=None,
                 filename='gillespie_test', fps=None, free=False, nAtt=None, nPullers=None, Ftot=None,
                 OrderParam=None, num_corners=36, grav_nonuniform=None, forceTotal=None, prob_rnd=None, kappa=None,
                 time_step=None, force_on_corner=None, rnd_corner=False):

        super().__init__(size=size, shape=shape, solver='gillespie', filename=filename, fps=fps, winner=winner,
                         position=position, angle=angle, frames=frames)

        self.free = free
        self.gillespie = None
        self.nAtt = nAtt
        self.nPullers = nPullers
        self.Ftot = Ftot
        self.OrderParam = OrderParam
        self.force = b2Vec2(0.0, 0.0)
        self.numCorner_with_force = np.random.choice([i for i in range(num_corners)])
        self.forceTotal = forceTotal
        self.prob_rnd = prob_rnd
        self.kappa = kappa
        self.time_step = time_step
        self.grav_nonuniform = grav_nonuniform
        self.force_on_corner = force_on_corner
        self.rnd_corner = rnd_corner
        self.theta = None

    def get_corners(self, my_maze):
        # fixtures = my_load.fixtures
        # vertices = [fixtures[i].shape.vertices for i in range(len(fixtures))]
        # vertices = [v for sublist in vertices for v in sublist]
        sites, _ = init_sites(my_maze, n=36, randonmize=False)

        return [my_maze.bodies[-1].GetWorldPoint(site) for site in sites]

    def get_direction_of_grav(self, my_maze, display=None):
        f_att_point = self.position[-1] if not self.force_on_corner else self.get_corners(my_maze)[self.numCorner_with_force]
        if self.grav_nonuniform and f_att_point[0] < my_maze.slits[0]:
            in_slit = [my_maze.slits[0], np.random.uniform(my_maze.arena_height / 2 - my_maze.exit_size / 2,
                                                           my_maze.arena_height / 2 + my_maze.exit_size / 2)]

            if display is not None:
                Point(in_slit, (100, 0, 0)).draw(display)
            to_grav_point = np.array(in_slit) - f_att_point
            to_grav_point_norm = to_grav_point / np.linalg.norm(to_grav_point)
            return b2Vec2(to_grav_point_norm[0], to_grav_point_norm[1])
        return b2Vec2(1, 0)

    def step_grav_simulation(self, my_maze, display=None):
        self.limit_vel(my_maze)
        self.forces(my_maze, display)
        my_maze.Step(self.time_step, 10, 10)
        self.position = np.vstack((self.position, [my_maze.bodies[-1].position.x, my_maze.bodies[-1].position.y]))
        self.angle = np.hstack((self.angle, my_maze.bodies[-1].angle))

    def randomize(self, my_maze, display=None):
        bias = self.get_direction_of_grav(my_maze, display)
        # angle of direction
        angle_bias = np.arctan2(bias[1], bias[0])

        self.theta = vonmises.rvs(self.kappa, loc=0, size=1)[0] + angle_bias
        if self.force_on_corner and self.rnd_corner:
            self.numCorner_with_force = \
                np.random.choice([i for i in range(len(self.get_corners(my_maze)))])

    def limit_vel(self, my_maze, limit_vel = 1):
        my_load = my_maze.bodies[-1]
        totalVel = np.linalg.norm(my_load.linearVelocity) + my_load.angularVelocity * my_maze.average_radius()
        if totalVel > 1:
            # print(totalVel)
            my_load.linearVelocity = my_load.linearVelocity / totalVel * limit_vel
            my_load.angularVelocity = my_load.angularVelocity / totalVel * limit_vel

    def forces(self, my_maze, display=None):
        if np.random.uniform(0, 1) < self.prob_rnd:
            self.randomize(my_maze, display)

        self.force = b2Vec2(np.cos(self.theta), np.sin(self.theta)) * self.forceTotal
        my_maze.bodies[-1].ApplyForce(self.force, self.get_att_force(my_maze), wake=True)

    def get_att_force(self, my_maze):
        f_att_point = self.get_corners(my_maze)[self.numCorner_with_force] \
            if self.force_on_corner else my_maze.bodies[-1].position
        return f_att_point

    def draw_force(self, my_maze, display):
        f_att_point = self.get_att_force(my_maze)
        Arrow(f_att_point,
              f_att_point + [self.force.x, self.force.y], 'puller').draw(display)

    def is_solved(self, my_maze):
        return self.position[-1][0] > my_maze.slits[0] + my_maze.slits[0] * 0.5

    def is_close_to_arenalength(self, my_maze):
        return self.position[-1][0] > my_maze.arena_length * 0.9

    def simulation_loop(self, my_maze, display=None):
        i = 0
        self.randomize(my_maze, display)
        while i < len(self.frames) - 1:
            self.step_grav_simulation(my_maze, display)
            i += 1
            end = self.is_close_to_arenalength(my_maze)
            if display is not None:
                end = display.update_screen(self, i) or end
                display.renew_screen(movie_name=self.filename)

            if end:
                self.frames = self.frames[:len(self.angle)]
                break

        if display is not None:
            display.end_screen()

    def run_gravitational_simulation(self, frameNumber, display=True):
        self.frames = np.linspace(1, frameNumber, frameNumber)
        angle = np.pi * 2 * np.random.uniform()
        my_maze = Maze(size=self.size, shape=self.shape, solver='gillespie', angle=angle,
                       geometry=self.geometry())
        my_maze.bodies[-1].linearDamping = 0
        my_maze.bodies[-1].angularDamping = 0

        self.position = np.array([[my_maze.arena_length / 4, my_maze.arena_height / 2]])
        self.angle = np.array([angle], dtype=float)  # array to store the position and angle of the load
        my_maze.set_configuration(self.position[0], self.angle[0])
        if display:
            display = Display(self.filename, self.fps, my_maze, wait=10)
        else:
            display = None
        self.simulation_loop(my_maze, display=display)

    def run_simulation(self, frameNumber):
        my_maze = Maze(size=self.size, shape=self.shape, solver='gillespie')
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
        if self.shape in ['H', 'I', 'T', 'LongI']:
            return 'MazeDimensions_ant.xlsx', 'LoadDimensions_ant.xlsx'
        elif self.shape == 'SPT':
            return 'MazeDimensions_new2021_SPT_ant_Amir.xlsx', 'LoadDimensions_new2021_SPT_ant_Amir.xlsx'
        else:
            raise ValueError(f'Unknown geometry for shape {self.shape}')

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


if __name__ == '__main__':
    x = Trajectory_gillespie(size='L',
                             shape='T',
                             filename='test',
                             forceTotal=10,
                             time_step=0.2,
                             prob_rnd=0.1,
                             kappa=5,
                             fps=1 / 0.2,
                             grav_nonuniform=True,
                             force_on_corner=True,
                             )
    x.run_gravitational_simulation(frameNumber=1000,
                                   display=True)