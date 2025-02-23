import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_SPACE)
from PhysicsEngine.drawables import colors
import os
import numpy as np
import sys
from directories import video_directory
import cv2
from datetime import datetime


class Display:
    def __init__(self, name: str, fps: int, my_maze, wait=0, cs=None, videowriter=False, config=None, ts=None,
                 frame=0, position=None, bias=None):
        self.my_maze = my_maze
        self.fps = fps
        self.name = name
        self.ppm = int(1100 / self.my_maze.arena_length)  # pixels per meter
        self.height = int(self.my_maze.arena_height * self.ppm)
        self.width = 1100

        pygame.font.init()  # display and fonts
        self.font = pygame.font.Font('freesansbold.ttf', 50)
        # self.monitor = {'left': 0, 'top': 0,
        #                 'width': int(Tk().winfo_screenwidth() * 0.9), 'height': int(Tk().winfo_screenheight() * 0.8)}
        self.monitor = {'left': 0, 'top': 0, 'width': self.width, 'height': self.height}
        self.screen = self.create_screen(position=position)
        self.arrows = []
        self.circles = []
        self.polygons = []
        self.points = []
        self.wait = wait
        self.i = frame
        self.ts = ts
        self.bias = bias
        if config is not None:
            my_maze.set_configuration(config[0], config[1])

        self.renew_screen(movie_name=name)
        self.cs = cs
        if self.cs is not None:
            self.scale_factor = {'Large': 1., 'Medium': 0.5, 'Small Far': 0.2,
                                 'Small Near': 0.2, 'Small': 0.2, 'M': 0.25, 'L': 0.5, 'XL': 1, 'S': 0.25}[
                self.my_maze.size]

        if videowriter:
            self.VideoShape = (self.monitor['height'], self.monitor['width'])
            self.VideoWriter = cv2.VideoWriter(self.video_directory(), cv2.VideoWriter_fourcc(*'DIVX'), 20,
                                               (self.VideoShape[1], self.VideoShape[0]))

    @staticmethod
    def video_directory():
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        path = os.path.join(video_directory, sys.argv[0].split('/')[-1].split('.')[0] + '_' + current_time + '.mp4v')
        print('Saving Movie in ', path)
        return path

    def create_screen(self, position=None) -> pygame.surface:
        pygame.font.init()  # display and fonts
        pygame.font.Font('freesansbold.ttf', 25)

        if self.my_maze.free:  # screen size dependent on trajectory_inheritance
            self.ppm = int(1000 / (np.max(position[:, 0]) - np.min(position[:, 0]) + 10))  # pixels per meter
            self.width = int((np.max(position[:, 0]) - np.min(position[:, 0]) + 10) * self.ppm)
            self.height = int((np.max(position[:, 1]) - np.min(position[:, 1]) + 10) * self.ppm)

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.monitor['left'], self.monitor['top'])
        screen = pygame.display.set_mode((self.width, self.height))
        if self.my_maze is not None:
            pygame.display.set_caption(self.my_maze.shape + ' ' + self.my_maze.size + ' ' + self.my_maze.solver)
        return screen

    def m_to_pixel(self, r):
        return [int(r[0] * self.ppm), self.height - int(r[1] * self.ppm)]

    def update_screen(self, x, i):
        self.i = i
        if self.wait > 0:
            pygame.time.wait(int(self.wait))
        self.draw(x)
        end = self.keyboard_events()
        return end

    def renew_screen(self, movie_name=None, frame_index=None, color_background=(250, 250, 250)):
        self.screen.fill(color_background)

        if self.bias is not None:
            # draw a thick line on the top of the screen in blue
            if self.bias == 'left':
                pygame.draw.line(self.screen, colors['bias'], [0, 10], [self.width, 10], 5)
            elif self.bias == 'right':
                pygame.draw.line(self.screen, colors['bias'], [0, self.height - 10], [self.width, self.height - 10], 5)

        self.drawGrid()
        self.polygons = self.circles = self.points = self.arrows = []

        text = self.font.render(movie_name, True, colors['text'])
        text_rect = text.get_rect()

        if frame_index is not None:
            text2 = self.font.render('Frame: ' + str(frame_index), True, colors['text'])
            self.screen.blit(text2, [0, 50])
            self.screen.blit(text, text_rect)

        if self.ts is not None:
            state = self.ts[self.i]
            text_state = self.font.render('state: ' + str(state), True, colors['text'])
            self.screen.blit(text_state, [0, 100])

    def end_screen(self):
        if hasattr(self, 'VideoWriter'):
            self.VideoWriter.release()
            print('Saved Movie in ', self.video_directory())
        pygame.display.quit()

    def pause_me(self):
        pygame.time.wait(int(100))
        events = pygame.event.get()
        for event in events:
            if event.type == KEYDOWN and event.key == K_SPACE:
                return
            if event.type == QUIT or \
                    (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.end_screen()
                return
        self.pause_me()

    def draw(self, x):
        self.my_maze.draw(self)
        if self.cs is not None:
            if self.i <= 1 or self.i >= len(x.angle) - 1:
                kwargs = {'color': (0, 0, 0), 'scale_factor': self.scale_factor}
            else:
                kwargs = {'scale_factor': self.scale_factor}
            self.cs.draw(x.position[self.i:self.i + 1], x.angle[self.i:self.i + 1], **kwargs)

        if hasattr(x, 'numCorner_with_force'):
            x.draw_force(self.my_maze, self)

        if hasattr(x, 'participants'):
            if hasattr(x.participants, 'forces'):
                x.participants.forces.draw(self, x)
            if hasattr(x.participants, 'positions'):
                x.participants.draw(self)
        self.display()

        if hasattr(self, 'VideoWriter'):
            self.write_to_Video()
        # self.write_to_Video()
        return

    def display(self):
        pygame.display.flip()

    def calc_fraction_of_circumference(self):
        img = np.swapaxes(pygame.surfarray.array3d(self.screen), 0, 1)
        shape_mask = np.logical_and(img[:, :, 0] == 250, img[:, :, 1] == 0)

        shift = 5
        size_borders = {'XL': [0, 472 + shift, 683 + shift, shape_mask.shape[1] - 1],
                        'L': [0, 474 + shift, 700 + shift, shape_mask.shape[1] - 1],
                        'M': [0, 450 + shift, 660 + shift, shape_mask.shape[1] - 1],
                        'S': [0, 491 + shift, 633 + shift, shape_mask.shape[1] - 1]}

        borders = size_borders[self.my_maze.size]

        fraction_of_circumference = np.zeros(shape=3)
        for i, (edge_left, edge_right) in enumerate(zip(borders[:-1], borders[1:])):
            chamber = np.zeros_like(shape_mask).astype(bool)
            chamber[:, edge_left:edge_right] = True

            fraction_of_circumference[i] = np.sum(np.logical_and(shape_mask, chamber)) / np.sum(shape_mask)
        return fraction_of_circumference

    def get_image(self):
        img = np.swapaxes(pygame.surfarray.array3d(self.screen), 0, 1)
        return img

    def write_to_Video(self):
        img = self.get_image()
        self.VideoWriter.write(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))

        # if self.ps is not None:
        #     self.ps.write_to_Video()

    def draw_contacts(self, contact):
        for contacts in contact:
            pygame.draw.circle(self.screen, colors['contact'],  # On the corner
                               [int(contacts[0] * self.ppm),
                                int(self.height - contacts[1] * self.ppm)],
                               10,
                               )

    def drawGrid(self):
        block = 2
        block_size = 2 * self.ppm
        for y in range(int(np.ceil(self.height / self.ppm / block) + 1)):
            for x in range(int(np.ceil(self.width / self.ppm / block))):
                rect = pygame.Rect(x * block_size, self.height -
                                   y * block_size, block_size, block_size)
                pygame.draw.rect(self.screen, colors['grid'], rect, 1)

    def keyboard_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT or \
                    (event.type == KEYDOWN and event.key == K_ESCAPE):  # you can also add 'or Finished'
                # The user closed the window or pressed escape
                self.end_screen()
                return True

            if event.type == KEYDOWN and event.key == K_SPACE:
                self.pause_me()
            # """
            # To control the frames:
            # 'D' = one frame forward
            # 'A' = one frame backward
            # '4' (one the keypad) = one second forward
            # '6' (one the keypad) = one second backward
            # """
            # if event.key == K_a:
            #     i -= 1
            # elif event.key == K_d:
            #     i += 1
            # elif event.key == K_KP4:
            #     i -= 30
            # elif event.key == K_KP6:
            #     i += 30

    def snapshot(self, filename, *args):
        pygame.image.save(self.screen, filename)
        if 'inlinePlotting' in args:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            fig.set_size_inches(30, 15)
            img = mpimg.imread(filename)
            plt.imshow(img)
        return
