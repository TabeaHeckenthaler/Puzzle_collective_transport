from abc import abstractmethod
from PhysicsEngine.drawables import Drawables, colors
import os
import pygame
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


class Participants(Drawables):
    def __init__(self, x, color=colors['puller']):
        super().__init__(color)
        self.solver = x.solver
        self.filename = x.filename
        self.frames = list()
        self.size = x.size
        self.VideoChain = x.VideoChain
        self.positions = np.array([])

    @abstractmethod
    def matlab_loading(self, x) -> None:
        pass

    @abstractmethod
    def averageCarrierNumber(self) -> float:
        pass

    def draw(self, display) -> None:
        fr_i = display.i-1  # frame index
        if self.solver == 'ant':
            for carrying in np.where(self.frames[fr_i].carrying.astype(bool))[0]:
                pygame.draw.circle(display.screen, (4, 188, 210),
                                   display.m_to_pixel(self.frames[fr_i].position[carrying]), 7.)

            for non_carrying in np.where(~self.frames[fr_i].carrying.astype(bool))[0]:
                pygame.draw.circle(display.screen, (225, 169, 132),
                                   display.m_to_pixel(self.frames[fr_i].position[non_carrying]), 7.)
        else:
            for part in range(self.positions[fr_i].shape[0]):
                pygame.draw.circle(display.screen, self.color,
                                   display.m_to_pixel(self.frames[fr_i].position[part, 0]), 7.)
