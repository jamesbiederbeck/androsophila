"""Flappy Bird boundary: raw rendered pixels in, one discrete button out.

Mirrors doom/game.py's Game class shape (pixels()/act()/observation()/close()/
new_episode()) so flappy/play.py can drive the same Brain/NativeBrain/GPUBrain
without any game-specific code outside this module and vision/retina.py.

A pipe pair in this game is really a hurdle, not a gate: the bird's x
position is fixed (`_player_x` never changes), so there is nothing to
dodge sideways, and it clears an obstacle the same way a hurdler does --
by gaining enough height, not by picking a lane. The stock sprite renders
the lower pipe as a narrow 52px column with open sky on both flanks, which
looks to a bilinear-sampled retina like a post with a way around it -- an
affordance the bird does not have. `pixels()` projects only the *lower*
pipe as a full-frame-width bar running from the ground up to the gap's
bottom edge (the hurdle height): solid across all of x, exactly the bar a
hurdler clears from below, infinitely wide because there is no lateral
axis to be finite in. The upper pipe is deliberately left unprojected --
a hurdle has no ceiling above the runner, and drawing one would reintroduce
the wrong intuition (squeezing through a slot) this change is meant to
remove. The upper pipe still exists in the underlying env and can still
end the episode on collision; only its *rendering* is skipped, so flying
too high remains just as fatal as before, it's simply not drawn as an
obstacle. This changes what is rendered for both the neural visual input
and the human broadcast view (they are always the same frame in this repo,
matching doom/game.py's convention) -- game physics, scoring and collision
are untouched; only the picture is reshaped to fit the bird's actual
constraints.
"""
import numpy as np
import pygame
from flappy_bird_gymnasium.envs.constants import PIPE_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv

# The env's only declared tick-rate signal (metadata['render_fps']); not
# enforced by the env itself unless render_mode='human', so this is a chosen
# real-time slice per env.step(), not a measured hardware clock the way
# ViZDoom's 35Hz is for doom/game.py.
FPS = 30


class Game:
    def __init__(self, seed=41027, score_limit=None, no_pipes=False, wall_pipes=True):
        self.env = FlappyBirdEnv(render_mode='rgb_array', use_lidar=False, score_limit=score_limit)
        if no_pipes:
            # Park every generated pipe far off both the visible screen and
            # the collision check, forever, instead of patching the vendored
            # submodule -- there is no constructor flag for this upstream.
            # Ground/ceiling boundaries are untouched and still terminate an
            # episode; only the pipe obstacle is removed.
            self.env._get_random_pipe = lambda: [{'x': -9999, 'y': -9999}, {'x': -9999, 'y': -9999}]
        self._wall_pipes = wall_pipes
        self._pipe_wall_color = None
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
        frame = self._frame.copy()
        if self._wall_pipes:
            hurdle_top = self._hurdle_height()
            if hurdle_top is not None:
                ground_y = int(self.env._ground['y'])
                if hurdle_top < ground_y: frame[hurdle_top:ground_y, :] = self._pipe_color()
        return frame

    def _hurdle_height(self):
        """Pixel row of the top edge of the nearest unpassed, on-screen
        lower pipe -- the hurdle bar the bird must clear from below. None
        if no such pipe exists (matches no_pipes and the brief window
        before the first pipe scrolls into view)."""
        env = self.env
        pairs = [(u, l) for u, l in zip(env._upper_pipes, env._lower_pipes)
                 if u['x'] < env._screen_width and u['x'] + PIPE_WIDTH > env._player_x]
        if not pairs: return None
        _, l = min(pairs, key=lambda pair: pair[0]['x'])
        return int(np.clip(l['y'], 0, env._screen_height))

    def _pipe_color(self):
        # Sampled once from the loaded sprite (not hardcoded) so this tracks
        # whatever --pipe-color the env was constructed with.
        if self._pipe_wall_color is None:
            sprite = pygame.surfarray.array3d(self.env._images['pipe'][0])
            self._pipe_wall_color = sprite[sprite.shape[0] // 2, sprite.shape[1] // 2].astype(np.uint8)
        return self._pipe_wall_color

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
