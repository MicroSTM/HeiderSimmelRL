from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import random
import math
import pygame
import Box2D
from Box2D.b2 import world, circleShape, edgeShape, dynamicBody
from Box2D import b2DistanceJointDef
from scipy.misc import imresize
from PIL import Image
import cv2


def _my_draw_edge(edge, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    vertices = [(body.transform * v) * PPM for v in edge.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.line(screen, color, vertices[0], vertices[1], 5)
edgeShape.draw = _my_draw_edge


def _my_draw_circle(circle, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(screen, color, [int(
        x) for x in position], int(circle.radius * PPM))
circleShape.draw = _my_draw_circle


def _my_draw_distance_joint(screen, circle1, body1, cirlce2, body2, color, PPM, SCREEN_HEIGHT):
    pos1 = body1.transform * circle1.pos * PPM
    pos1 = (pos1[0], SCREEN_HEIGHT - pos1[1])
    pos2 = body2.transform * cirlce2.pos * PPM
    pos2 = (pos2[0], SCREEN_HEIGHT - pos2[1])
    pygame.draw.line(screen, color, [int(x) for x in pos1], [int(x) for x in pos2], 3)


def _get_world_pos(body):
    """get the position"""
    position = body.transform * body.fixtures[0].shape.pos
    return position


def _get_obs(screen, SCREEN_WIDTH, SCREEN_HEIGHT):
    string_image = pygame.image.tostring(screen, 'RGB')
    temp_surf = pygame.image.fromstring(string_image, 
                    (SCREEN_WIDTH, SCREEN_HEIGHT),'RGB')
    return(pygame.surfarray.array3d(temp_surf))


def _get_dist(pos1, pos2):
    """get the distance between two 2D positions"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


class collision:
    """Two individual objects"""
    def __init__(self, PPM=20.0, 
                       TARGET_FPS=60,
                       SCREEN_WIDTH=640,
                       SCREEN_HEIGHT=480,
                       TIME_STEP=5,
                       enable_renderer=True):
        self.PPM = PPM
        self.TARGET_FPS = 60
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.nb_time_steps = TIME_STEP
        self.TIME_STEP = 1.0 / TARGET_FPS * TIME_STEP
        self.enable_renderer = enable_renderer

        if self.enable_renderer:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Blocking_v0')
            self.clock = pygame.time.Clock()
        self.room_dim = (16, 16)
        self.door_length = 3

        random.seed(1)


    def setup(self, radius=[1, 1], density=[1, 1], record_path=None):
        """setup a new espisode"""
        self.world = world(gravity=(0, 0), doSleep=True)
        self.room = self.world.CreateBody(position=(16, 12))
        self.room.CreateEdgeChain(
            [
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2 - self.door_length)]
        )
        self.agents = []
        # object 1
        x, y = 16 + random.uniform(-6, 6), 12 + random.uniform(-6, 6)
        body = self.world.CreateDynamicBody(position=(x, y))
        body.CreateCircleFixture(radius=1, density=density[0], friction=0.3, restitution = 1)
        self.agents.append(body)
        self.trajectories = [[(x, y)]]
        # object 2
        while True:
            x2, y2 = 16 + random.uniform(-6, 6), 12 + random.uniform(-6, 6)
            if _get_dist((x, y), (x2, y2)) > 2.0: # no collision
                break
        body = self.world.CreateDynamicBody(position=(x2, y2))
        body.CreateCircleFixture(radius=1, density=density[1], friction=0.3, restitution = 1)
        self.agents.append(body)
        self.trajectories.append([(x2, y2)])
        self.steps = 0
        self.running = False
        self.video = None
        if record_path:
            self.video = cv2.VideoWriter(
                record_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                20 * 5, 
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))


    def start(self):
        """start the episode"""
        self.running = True
        for t in range(self.nb_time_steps):
            self.world.Step(1.0 / self.TARGET_FPS, 10, 10)
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]


    def apply_force(self, agent_id, fx, fy):
        """apply force"""
        f = self.agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.agents[agent_id].GetWorldPoint(localPoint=(0.0, 0.0))
        self.agents[agent_id].ApplyForce(f, p, True)


    def step(self):
        """apply one step and update the environment"""
        self.steps += 1
        for t in range(self.nb_time_steps):
            self.world.Step(1.0 / self.TARGET_FPS, 10, 10)
        self.world.ClearForces()
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]
        for agent_id, agent_pos in enumerate(self.agents_pos):
            self.trajectories[agent_id].append((agent_pos[0], agent_pos[1]))
        if self.enable_renderer:
            self.render()


    def render(self):
        """render the environment"""
        colors = {
            0: (0, 0, 0, 255), # ground body
            1: (255, 0, 0, 255), # agent 1
            2: (0, 255, 0, 255), # agent 2
        }
        self.screen.fill((255, 255, 255, 255))
        for body_id, body in enumerate([self.room, self.agents[0], self.agents[1]]):
            for fixture in body.fixtures:
                fixture.shape.draw(self.screen, body, fixture, colors[body_id], self.PPM, self.SCREEN_HEIGHT)
        pygame.display.flip()
        self.clock.tick(self.TARGET_FPS)
        if self.video:
            obs = _get_obs(self.screen, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            self.video.write(cv2.cvtColor(obs.transpose(1, 0, 2), 
                                            cv2.COLOR_RGB2BGR))


    def release(self):
        """release video writer"""
        if self.video:
            # self.step()
            self.video.release()
            self.video = None


    def destroy(self):
        """destroy the environment"""
        self.world.DestroyBody(self.room)
        self.room = None
        for agent in self.agents:
            self.world.DestroyBody(agent)
        self.agents = []
        if self.enable_renderer:
            pygame.quit()


class rod:
    """Two connected objects"""
    def __init__(self, PPM=20.0, 
                       TARGET_FPS=60,
                       SCREEN_WIDTH=640,
                       SCREEN_HEIGHT=480,
                       TIME_STEP=5,
                       enable_renderer=True):
        self.PPM = PPM
        self.TARGET_FPS = 60
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.nb_time_steps = TIME_STEP
        self.TIME_STEP = 1.0 / TARGET_FPS * TIME_STEP
        self.enable_renderer = enable_renderer

        if self.enable_renderer:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('collision')
            self.clock = pygame.time.Clock()
        self.room_dim = (16, 16)
        self.door_length = 3
        self.f, self.d = 4, 0.5

        random.seed(1)


    def setup(self, radius=[1, 1], density=[1, 1], record_path=None):
        """setup a new espisode"""
        self.world = world(gravity=(0, 0), doSleep=True)
        self.room = self.world.CreateBody(position=(16, 12))
        self.room.CreateEdgeChain(
            [
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2 - self.door_length)]
        )
        self.agents = []
        # object 1
        x, y = 16 + random.uniform(-6, 6), 12 + random.uniform(-6, 6)
        body = self.world.CreateDynamicBody(position=(x, y))
        body.CreateCircleFixture(radius=1, density=density[0], friction=0.3, restitution = 1)
        self.agents.append(body)
        self.trajectories = [[(x, y)]]
        # object 2
        while True:
            x2, y2 = 16 + random.uniform(-6, 6), 12 + random.uniform(-6, 6)
            if _get_dist((x, y), (x2, y2)) > 2.0: # no collision
                break
        body = self.world.CreateDynamicBody(position=(x2, y2))
        body.CreateCircleFixture(radius=1, density=density[1], friction=0.3, restitution = 1)
        self.agents.append(body)
        self.trajectories.append([(x2, y2)])
        
        dfn = b2DistanceJointDef(
                frequencyHz=self.f,
                dampingRatio=self.d,
                bodyA=self.agents[0],
                bodyB=self.agents[1],
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
            )
        self.joint = self.world.CreateJoint(dfn)

        self.steps = 0
        self.running = False
        self.video = None
        if record_path:
            self.video = cv2.VideoWriter(
                record_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                20 * 5, 
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))


    def start(self):
        """start the episode"""
        self.running = True
        for t in range(self.nb_time_steps):
            self.world.Step(1.0 / self.TARGET_FPS, 10, 10)
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]


    def apply_force(self, agent_id, fx, fy):
        """apply force"""
        f = self.agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.agents[agent_id].GetWorldPoint(localPoint=(0.0, 0.0))
        self.agents[agent_id].ApplyForce(f, p, True)


    def step(self):
        """apply one step and update the environment"""
        self.steps += 1
        for t in range(self.nb_time_steps):
            self.world.Step(1.0 / self.TARGET_FPS, 10, 10)
        self.world.ClearForces()
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]
        for agent_id, agent_pos in enumerate(self.agents_pos):
            self.trajectories[agent_id].append((agent_pos[0], agent_pos[1]))
        self.render()


    def get_obs(self):
        """get observation"""
        return imresize(_get_obs(self.screen, 
                        self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 
                        (self.obs_dim[1], self.obs_dim[2])).transpose([2, 0, 1]).reshape((-1))


    def get_reward(self):
        """get reward"""
        return -1.0 if not _in_room(self.agents[0], self.room) else 0.0 


    def render(self):
        """render the environment"""
        colors = {
            0: (0, 0, 0, 255), # ground body
            1: (255, 0, 0, 255), # agent 1
            2: (0, 255, 0, 255), # agent 2
        }
        self.screen.fill((255, 255, 255, 255))
        circles = []
        for body_id, body in enumerate([self.room, self.agents[0], self.agents[1]]):
            for fixture in body.fixtures:
                fixture.shape.draw(self.screen, body, fixture, colors[body_id], self.PPM, self.SCREEN_HEIGHT)
                if body_id:
                    circles.append(fixture.shape)
        pygame.display.flip()
        self.clock.tick(self.TARGET_FPS)
        if self.video:
            obs = _get_obs(self.screen, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            self.video.write(cv2.cvtColor(obs.transpose(1, 0, 2), 
                                            cv2.COLOR_RGB2BGR))


    def release(self):
        """release video writer"""
        if self.video:
            self.video.release()
            self.video = None


    def destroy(self):
        """destroy the environment"""
        self.world.DestroyBody(self.room)
        self.room = None
        for agent in self.agents:
            self.world.DestroyBody(agent)
        self.joint = None
        self.agents = []
        if self.enable_renderer:
            pygame.quit()


class rope(rod):
    def __init__(self, PPM=20.0, 
                       TARGET_FPS=60,
                       SCREEN_WIDTH=640,
                       SCREEN_HEIGHT=480,
                       TIME_STEP=5,
                       enable_renderer=True):
        super(rope, self).__init__(PPM, TARGET_FPS, SCREEN_WIDTH, SCREEN_HEIGHT, TIME_STEP, enable_renderer)
        if self.enable_renderer:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('rope')
            self.clock = pygame.time.Clock()
        self.f, self.d = 0.1, 0


class spring(rod):
    def __init__(self, PPM=20.0, 
                       TARGET_FPS=60,
                       SCREEN_WIDTH=640,
                       SCREEN_HEIGHT=480,
                       TIME_STEP=5,
                       enable_renderer=True):
        super(spring, self).__init__(PPM, TARGET_FPS, SCREEN_WIDTH, SCREEN_HEIGHT, TIME_STEP, enable_renderer)
        if self.enable_renderer:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('spring')
            self.clock = pygame.time.Clock()
        self.f, self.d = 0.3, 0
