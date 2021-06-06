from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import random
import pygame
import Box2D
from Box2D.b2 import world, circleShape, edgeShape, dynamicBody
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


def _get_world_pos(body):
    """get the position"""
    position = body.transform * body.fixtures[0].shape.pos
    return position


def _get_pos(body):
    """get the position"""
    position = body.transform * body.fixtures[0].shape.pos
    return position


def _get_body_bound(body):
    """get the boundary of a cicle"""
    position = body.transform * body.fixtures[0].shape.pos
    radius = body.fixtures[0].shape.radius
    return (position[0] - radius, position[1] - radius,
            position[0] + radius, position[1] + radius)


def _get_door(body):
    """get the door's location"""
    vertices1 = [(body.transform * v)  \
                    for v in body.fixtures[0].shape.vertices]
    vertices2 = [(body.transform * v) \
                    for v in body.fixtures[-1].shape.vertices]

    return [vertices1[0], vertices2[-1]]


def _get_room_bound(body):
    """get the boundary of a room (upper-left corner + bottom-right corner)"""
    x_list, y_list = [], []
    for fixture in body.fixtures:
        vertices = [(body.transform * v) \
                    for v in fixture.shape.vertices]
        x_list += [v[0] for v in vertices]
        y_list += [v[1] for v in vertices]
    min_x, min_y = min(x_list), min(y_list)
    max_x, max_y = max(x_list), max(y_list)
    return (min_x, min_y, max_x, max_y)


def _in_room(body, room):
    body_bound = _get_body_bound(body)
    min_x, min_y, max_x, max_y = _get_room_bound(room)
    return body_bound[2] >= min_x and body_bound[3] >= min_y and \
           body_bound[0] <= max_x and body_bound[1] <= max_y


def _get_obs(screen, SCREEN_WIDTH, SCREEN_HEIGHT):
    string_image = pygame.image.tostring(screen, 'RGB')
    temp_surf = pygame.image.fromstring(string_image, 
                    (SCREEN_WIDTH, SCREEN_HEIGHT),'RGB')
    return(pygame.surfarray.array3d(temp_surf))


class Blocking_v0:
    """agent 2 blocks object 1, discrete action"""
    def __init__(self, PPM=20.0, 
                       TARGET_FPS=60,
                       SCREEN_WIDTH=640,
                       SCREEN_HEIGHT=480,
                       TIME_STEP=5,
                       enable_renderer=True,
                       random_pos=False,
                       restitution=0):
        self.PPM = PPM
        self.TARGET_FPS = 60
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.TIME_STEP = 1.0 / TARGET_FPS * TIME_STEP
        self.enable_renderer = enable_renderer
        self.obs_dim = (3, 86, 86)

        self.action_space = ['up', 'down', 'left', 'right', 'upleft', 'upright', 'downleft', 'downright', 'stop']
        self.action_size = len(self.action_space)
        if self.enable_renderer:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Blocking_v0')
            self.clock = pygame.time.Clock()
        self.room_dim = (16, 16)
        self.door_length = 3
        self.agent1_pos_list = [(16 + 7, 12 - 7), (16 - 7, 12 - 7), (16 + 7, 12 + 7)]
        self.agent2_pos_list = [(10, 6), (22, 6), (22, 18), (10, 18)]
        self.random_pos = random_pos
        self.restitution = restitution

        random.seed(1)


    def setup(self, record_path=None):
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
        # agent 1
        if self.random_pos:
            x, y = random.choice(self.agent1_pos_list)
            body = self.world.CreateDynamicBody(position=(x, y))
            body.CreateCircleFixture(radius=1, density=1, friction=0.3, restitution=self.restitution)
            self.agents.append(body)
            self.trajectories = [[(x, y)]]
        else:
            body = self.world.CreateDynamicBody(position=(16 + 4, 12 - 4))
            body.CreateCircleFixture(radius=1, density=1, friction=0.3, restitution=self.restitution)
            self.agents.append(body)
            self.trajectories = [[(16 + 4, 12 - 4)]]
        # agent 2
        x, y = random.choice(self.agent2_pos_list)
        body = self.world.CreateDynamicBody(position=(x, y))
        body.CreateCircleFixture(radius=1, density=1, friction=0.3)
        self.agents.append(body)
        self.trajectories.append([(x, y)])
        self.steps = 0
        self.running = False
        self.video = None
        if record_path:
            self.video = cv2.VideoWriter(
                record_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                20, 
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))


    def start(self):
        """start the episode"""
        self.running = True
        fx, fy = -1200.0, 1200.0
        f = self.agents[0].GetWorldVector(localVector=(fx, fy))
        p = self.agents[0].GetWorldPoint(localPoint=(0.0, 0.0))
        self.agents[0].ApplyForce(f, p, True)
        self.world.Step(1.0 / self.TARGET_FPS, 10, 10)
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]


    def send_action(self, agent_id, action):
        """send action to an agent"""
        fx, fy = 0.0, 0.0
        df = 300.0
        if action == 'up':
            fy += df
        elif action == 'down':
            fy -= df
        elif action == 'left':
            fx -= df
        elif action == 'right':
            fx += df
        elif action == 'upleft':
            fx -= df
            fy += df
        elif action == 'upright':
            fx += df
            fy += df
        elif action == 'downleft':
            fx -= df
            fy -= df
        elif action == 'downright':
            fx += df
            fy -= df
        elif action == 'stop':
            x, y = self.agents_pos[agent_id]
            self.world.DestroyBody(self.agents[agent_id])
            self.agents[agent_id] = self.world.CreateDynamicBody(position=(x, y), angle=0)
            if agent_id == 0:
                self.agents[agent_id].CreateCircleFixture(radius=1, density=1, friction=0.3, restitution=self.restitution)
            else:
                self.agents[agent_id].CreateCircleFixture(radius=1, density=1, friction=0.3)
            return
        else:
            print('ERROR: invalid action!')
        f = self.agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.agents[agent_id].GetWorldPoint(localPoint=(0.0, 0.0))
        self.agents[agent_id].ApplyForce(f, p, True)


    def step(self):
        """apply one step and update the environment"""
        self.steps += 1
        self.world.Step(self.TIME_STEP, 10, 10)
        self.world.ClearForces()
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]
        for agent_id, agent_pos in enumerate(self.agents_pos):
            self.trajectories[agent_id].append((agent_pos[0], agent_pos[1]))
        self.running = not self.terminal()
        self.render()


    def get_obs(self):
        """get observation"""
        return [imresize(_get_obs(self.screen, 
                        self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 
                        (self.obs_dim[1], self.obs_dim[2])).transpose([2, 0, 1]).reshape((-1))
                for _ in range(2)]


    def terminal(self):
        """check if the goal is achieved"""
        return not _in_room(self.agents[0], self.room)


    def get_reward(self):
        return [1.0, -1.0] if not _in_room(self.agents[0], self.room) else [0.0, 0.0]


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


class Blocking_v1(Blocking_v0):
    """agent 2 blocks agent 1, discrete action"""
    def __init__(self, PPM=20.0, 
                       TARGET_FPS=60,
                       SCREEN_WIDTH=640,
                       SCREEN_HEIGHT=480,
                       TIME_STEP=5,
                       enable_renderer=True,
                       random_pos=False,
                       restitution=0):
        super(Blocking_v1, self).__init__(PPM, TARGET_FPS, SCREEN_WIDTH, SCREEN_HEIGHT, TIME_STEP, enable_renderer, random_pos, restitution)
        if self.enable_renderer:
            pygame.display.set_caption('Blocking_v1')


    def start(self):
        """start the episode"""
        self.running = True
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]


class Blocking_v2(Blocking_v1):
    """agent 2 blocks agent 1 (can not stop), discrete action"""
    def __init__(self, PPM=20.0, 
                       TARGET_FPS=60,
                       SCREEN_WIDTH=640,
                       SCREEN_HEIGHT=480,
                       TIME_STEP=5,
                       enable_renderer=True,
                       random_pos=False,
                       restitution=0):
        super(Blocking_v2, self).__init__(PPM, TARGET_FPS, SCREEN_WIDTH, SCREEN_HEIGHT, TIME_STEP, enable_renderer, random_pos, restitution)
        if self.enable_renderer:
            pygame.display.set_caption('Blocking_v2')


    def send_action(self, agent_id, action):
        """send action to an agent"""
        fx, fy = 0.0, 0.0
        df = 300.0
        if action == 'up':
            fy += df
        elif action == 'down':
            fy -= df
        elif action == 'left':
            fx -= df
        elif action == 'right':
            fx += df
        elif action == 'upleft':
            fx -= df
            fy += df
        elif action == 'upright':
            fx += df
            fy += df
        elif action == 'downleft':
            fx -= df
            fy -= df
        elif action == 'downright':
            fx += df
            fy -= df
        elif action == 'stop':
            if agent_id == 1:
                x, y = self.agents_pos[agent_id]
                self.world.DestroyBody(self.agents[agent_id])
                self.agents[agent_id] = self.world.CreateDynamicBody(position=(x, y), angle=0)
                self.agents[agent_id].CreateCircleFixture(radius=1, density=1, friction=0.3)
            return
        else:
            print('ERROR: invalid action!')
        f = self.agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.agents[agent_id].GetWorldPoint(localPoint=(0.0, 0.0))
        self.agents[agent_id].ApplyForce(f, p, True)

