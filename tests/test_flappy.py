"""Meaningful causal/numerical boundaries for the Flappy Bird harness, not a
biological validation. Mirrors tests/test_doom.py's coverage shape."""
import numpy as np
from flappy.controls import FlapControls
from vision.retina import BilinearLuminance


def test_bilinear_luminance_wraps_retinal_samples_unchanged():
    from doom.game import retinal_samples
    uv = np.array([[0, 0], [1, 1], [.5, .5]], dtype=np.float32)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0, 0] = 255
    np.testing.assert_array_equal(BilinearLuminance().sample(image, uv), retinal_samples(image, uv))


def test_flap_controls_do_not_auto_fire():
    rows = [{'index': 0, 'id': '1', 'type': 'DLMn', 'side': 'L'}, {'index': 1, 'id': '2', 'type': 'DLMn', 'side': 'R'},
            {'index': 2, 'id': '3', 'type': 'DNa02', 'side': 'L'}]
    decoder = FlapControls(rows)
    a = decoder.decode(np.zeros(3), .1)
    assert not a['flap']
    a = decoder.decode(np.array([1, 0, 5]), .1)
    assert a['flap']  # DLMn (left) spiked
    a = decoder.decode(np.zeros(3), .1)
    assert not a['flap']  # no fire from lingering filtered activity
    a = decoder.decode(np.array([0, 2, 0]), .1)
    assert a['flap']  # either side triggers, matching doom's attack (no side filter)


def test_flap_controls_only_trigger_on_configured_readout_type():
    rows = [{'index': 0, 'id': '1', 'type': 'DNa02', 'side': 'L'}]
    decoder = FlapControls(rows)
    a = decoder.decode(np.array([5]), .1)
    assert not a['flap']  # DNa02 spiking is not DLMn; no flap


def test_game_pixels_and_act_advance_tick_deterministically():
    from flappy.game import Game
    g = Game(seed=41027)
    try:
        assert g.pixels().shape == (512, 288, 3)
        assert g.pixels().dtype == np.uint8
        assert g.observation()['tick'] == 0
        g.act(False)
        assert g.observation()['tick'] == 1
    finally:
        g.close()


def test_lower_pipe_is_projected_as_a_full_width_hurdle():
    from flappy.game import Game
    g = Game(seed=41027)
    try:
        for _ in range(60):
            g.act(False)
            if g.observation()['finished']:
                g.new_episode()
        hurdle_top = g._hurdle_height()
        assert hurdle_top is not None
        frame = g.pixels()
        color = g._pipe_color()
        ground_y = int(g.env._ground['y'])
        # solid, full-width, single-color bar from the hurdle height to the ground
        np.testing.assert_array_equal(frame[hurdle_top], np.tile(color, (frame.shape[1], 1)))
        np.testing.assert_array_equal(frame[ground_y - 1], np.tile(color, (frame.shape[1], 1)))
        # above the hurdle is untouched (no ceiling band -- a hurdle has none)
        assert not np.array_equal(frame[max(hurdle_top - 5, 0), 0], color)
    finally:
        g.close()


def test_wall_pipes_can_be_disabled_to_get_the_raw_frame():
    from flappy.game import Game
    g = Game(seed=41027, wall_pipes=False)
    try:
        for _ in range(60):
            g.act(False)
            if g.observation()['finished']:
                g.new_episode()
        np.testing.assert_array_equal(g.pixels(), g._frame)
    finally:
        g.close()


def test_no_pipes_leaves_wall_projection_a_no_op():
    from flappy.game import Game
    g = Game(seed=41027, no_pipes=True)
    try:
        assert g._hurdle_height() is None
        np.testing.assert_array_equal(g.pixels(), g._frame)
    finally:
        g.close()


def test_game_is_seed_reproducible():
    from flappy.game import Game
    g1 = Game(seed=7)
    g2 = Game(seed=7)
    try:
        np.testing.assert_array_equal(g1.pixels(), g2.pixels())
        for flap in [False, False, True, False, False]:
            g1.act(flap)
            g2.act(flap)
        np.testing.assert_array_equal(g1.pixels(), g2.pixels())
        assert g1.observation() == g2.observation()
    finally:
        g1.close()
        g2.close()
