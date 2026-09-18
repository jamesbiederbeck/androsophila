"""Flappy Bird boundary: raw rendered pixels in, one discrete button out.

Mirrors doom/game.py's Game class shape (pixels()/act()/observation()/close()/
new_episode()) so flappy/play.py can drive the same Brain/NativeBrain/GPUBrain
without any game-specific code outside this module and vision/retina.py.

`pixels()` does not return the game's own side-on frame. It returns the
world reprojected into the bird's first-person view (flappy/render3d.py):
each pipe becomes a wall standing on the ground, spanning the full lateral
axis and rising to the gap's lower edge -- a hurdle crossing the view
horizontally, placed by distance rather than by screen position. See that
module for why the shipped camera's picture misrepresents what the bird can
actually do about an obstacle.

Only the *lower* pipe is projected: a hurdle has no ceiling above the
runner. The upper pipe still exists in the env and still ends the episode
on collision -- flying too high remains exactly as fatal -- it is simply
not drawn as an obstacle. Game physics, scoring and collision are untouched
throughout; only the picture changes, for both the neural visual input and
the human broadcast view (always the same frame here, matching
doom/game.py's convention).
"""
import numpy as np
import pygame
from flappy_bird_gymnasium.envs.constants import PIPE_VEL_X, PLAYER_HEIGHT, PLAYER_WIDTH
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
from flappy.render3d import MIN_DEPTH, render

# The env's only declared tick-rate signal (metadata['render_fps']); not
# enforced by the env itself unless render_mode='human', so this is a chosen
# real-time slice per env.step(), not a measured hardware clock the way
# ViZDoom's 35Hz is for doom/game.py.
FPS = 30


class Game:
    def __init__(self, seed=41027, score_limit=None, no_pipes=False, perspective=True):
        self.env = FlappyBirdEnv(render_mode='rgb_array', use_lidar=False, score_limit=score_limit)
        if no_pipes:
            # Park every generated pipe far off both the visible screen and
            # the collision check, forever, instead of patching the vendored
            # submodule -- there is no constructor flag for this upstream.
            # Ground/ceiling boundaries are untouched and still terminate an
            # episode; only the pipe obstacle is removed.
            self.env._get_random_pipe = lambda: [{'x': -9999, 'y': -9999}, {'x': -9999, 'y': -9999}]
        self._no_pipes = no_pipes
        self._perspective = perspective
        self._colors = None
        self._seed = seed
        self.episode = 0
        self.new_episode()

    def new_episode(self):
        # Seed only the first reset (matches doom/game.py's Game, which sets
        # the ViZDoom seed once at construction); later resets continue the
        # same RNG stream rather than replaying identical episodes forever.
        self.env.reset(seed=self._seed if self.episode == 0 else None)
        self._frame = self.env.render()
        self.episode += 1
        self.tick = 0
        self._finished = False
        self._score = 0

    def pixels(self):
        if self._finished: raise RuntimeError('Episode finished; reset is required')
        if not self._perspective: return self._frame.copy()
        env = self.env
        return render((env._screen_height, env._screen_width),
                      float(env._player_y) + PLAYER_HEIGHT / 2, float(env._ground['y']),
                      self._hurdles(), self.tick * abs(PIPE_VEL_X), self._palette())

    def _hurdles(self):
        """(depth ahead, top edge y) for every lower pipe still in front of
        the bird -- the walls it has to clear."""
        # reset() keeps its own x values for the opening pipes and only takes
        # y from _get_random_pipe, so the no_pipes parking trick leaves them
        # at real depths with a sentinel height -- which would project as a
        # wall taller than the world. There is nothing to clear; say so.
        if self._no_pipes: return []
        player_x = self.env._player_x
        return [(float(pipe['x']) - player_x, float(pipe['y'])) for pipe in self.env._lower_pipes
                if float(pipe['x']) - player_x >= MIN_DEPTH]

    def _palette(self):
        # Sampled once from the game's own sprites and background rather than
        # hardcoded, so the view tracks whatever colors the env was built with.
        if self._colors is None:
            pipe = pygame.surfarray.array3d(self.env._images['pipe'][0])
            ground = self._frame[min(int(self.env._ground['y']) + 20, self._frame.shape[0] - 1), 5]
            self._colors = {
                'sky': self._frame[0, 0].astype(np.uint8),
                'hurdle': pipe[pipe.shape[0] // 2, pipe.shape[1] // 2].astype(np.uint8),
                'ground': ground.astype(np.uint8),
                'stripe': (ground * .82).astype(np.uint8),
            }
        return self._colors

    def act(self, flap):
        # The adapter is the only caller of env.step. No human keystrokes.
        _, reward, terminated, truncated, info = self.env.step(int(bool(flap)))
        self.tick += 1
        self._finished = terminated or truncated
        self._score = info['score']
        if not self._finished:
            self._frame = self.env.render()
        return float(reward)

    def observation(self):
        # _player_y/_player_vel_y/_player_rot are private env attributes (no
        # public accessor exists); vel_y positive = falling, negative = rising,
        # range [PLAYER_MIN_VEL_Y, PLAYER_MAX_VEL_Y] per
        # flappy_bird_gymnasium.envs.constants.
        return {'episode': self.episode, 'tick': self.tick, 'finished': self._finished,
                'score': self._score, 'y_velocity': float(self.env._player_vel_y),
                'y': float(self.env._player_y), 'rotation': float(self.env._player_rot)}

    def geometry(self):
        """Fixed screen/sprite dimensions for a coordinate-only renderer (no
        video frame) to draw the bird itself; static for the env's lifetime."""
        return {'screen_width': self.env._screen_width, 'screen_height': self.env._screen_height,
                'ground_y': self.env._ground['y'], 'player_x': self.env._player_x,
                'player_width': PLAYER_WIDTH, 'player_height': PLAYER_HEIGHT}

    def close(self):
        self.env.close()
