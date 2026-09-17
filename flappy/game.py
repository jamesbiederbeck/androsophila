"""Flappy Bird boundary: raw rendered pixels in, one discrete button out.

Mirrors doom/game.py's Game class shape (pixels()/act()/observation()/close()/
new_episode()) so flappy/play.py can drive the same Brain/NativeBrain/GPUBrain
without any game-specific code outside this module and vision/retina.py.
"""
from flappy_bird_gymnasium.envs.constants import PLAYER_HEIGHT, PLAYER_WIDTH
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv

# The env's only declared tick-rate signal (metadata['render_fps']); not
# enforced by the env itself unless render_mode='human', so this is a chosen
# real-time slice per env.step(), not a measured hardware clock the way
# ViZDoom's 35Hz is for doom/game.py.
FPS = 30


class Game:
    def __init__(self, seed=41027, score_limit=None, no_pipes=False):
        self.env = FlappyBirdEnv(render_mode='rgb_array', use_lidar=False, score_limit=score_limit)
        if no_pipes:
            # Park every generated pipe far off both the visible screen and
            # the collision check, forever, instead of patching the vendored
            # submodule -- there is no constructor flag for this upstream.
            # Ground/ceiling boundaries are untouched and still terminate an
            # episode; only the pipe obstacle is removed.
            self.env._get_random_pipe = lambda: [{'x': -9999, 'y': -9999}, {'x': -9999, 'y': -9999}]
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
        return self._frame.copy()

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
