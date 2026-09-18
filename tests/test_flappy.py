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


def test_pixels_are_the_first_person_view_not_the_game_camera():
    from flappy.game import Game
    g = Game(seed=41027)
    try:
        for t in range(18):
            g.act(t % 6 == 0)
        frame = g.pixels()
        assert frame.shape == (512, 288, 3)
        # every row is one flat color across: the world here is only ground,
        # sky and laterally-infinite walls, none of which vary left to right
        for row in range(0, 512, 32):
            assert len(np.unique(frame[row], axis=0)) == 1, row
        assert not np.array_equal(frame, g._frame)  # not the side-on frame
    finally:
        g.close()


def test_hurdles_are_the_lower_pipes_ahead_of_the_bird():
    from flappy.game import Game
    g = Game(seed=41027)
    try:
        for _ in range(20):
            g.act(False)
        hurdles = g._hurdles()
        assert hurdles
        assert all(depth > 0 for depth, _ in hurdles)  # only what is still ahead
        tops = {top for _, top in hurdles}
        assert tops == {float(p['y']) for p in g.env._lower_pipes
                        if float(p['x']) - g.env._player_x > 0}
        # approaching: every wall gets nearer as the bird flies on
        before = sorted(d for d, _ in hurdles)
        g.act(False)
        assert sorted(d for d, _ in g._hurdles())[0] < before[0]
    finally:
        g.close()


def test_perspective_can_be_disabled_to_get_the_raw_game_frame():
    from flappy.game import Game
    g = Game(seed=41027, perspective=False)
    try:
        for _ in range(20):
            g.act(False)
        np.testing.assert_array_equal(g.pixels(), g._frame)
    finally:
        g.close()


def test_no_pipes_leaves_an_empty_world_to_fly_through():
    from flappy.game import Game
    g = Game(seed=41027, no_pipes=True)
    try:
        assert g._hurdles() == []
        frame = g.pixels()
        # sky above the horizon, ground below, no wall anywhere
        assert len(np.unique(frame[:256].reshape(-1, 3), axis=0)) == 1
        assert not np.array_equal(frame[300, 0], frame[0, 0])
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
