"""Tests for the reference-orbit cache (real-time exploration speedup)."""

import numpy as np

from fractalforge.engine.precision import (
    clear_orbit_cache,
    compute_reference_orbit,
)

SEAHORSE_RE = "-0.7436438870371587"
SEAHORSE_IM = "0.1318259043091895"


def setup_function(_fn):
    clear_orbit_cache()


def test_cache_hit_same_request():
    a = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=800, zoom=1e20)
    b = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=800, zoom=1e20)
    assert a is b


def test_cache_hit_lower_precision_request():
    a = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=800, zoom=1e30)
    b = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=800, zoom=1e20)
    assert a is b  # higher-precision orbit satisfies a shallower request


def test_cache_miss_higher_precision():
    a = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=800, zoom=1e20)
    b = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=800, zoom=1e60)
    assert a is not b
    assert b.precision > a.precision


def test_cache_escaped_orbit_covers_larger_max_iter():
    # This seahorse point escapes; a complete (escaped) orbit satisfies any
    # larger iteration request.
    a = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=5000, zoom=1e20)
    assert a.escape_iter >= 0
    b = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=9000, zoom=1e20)
    assert a is b


def test_cache_nonescaped_orbit_recomputes_for_more_iters():
    # c = i is bounded (Misiurewicz): orbit never escapes, so a larger
    # iteration budget must recompute.
    a = compute_reference_orbit("0.0", "1.0", max_iter=500, zoom=1e20)
    assert a.escape_iter < 0
    b = compute_reference_orbit("0.0", "1.0", max_iter=1000, zoom=1e20)
    assert a is not b
    assert b.num_iters == 1000


def test_cache_distinct_centers():
    a = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=500, zoom=1e20)
    b = compute_reference_orbit("0.0", "1.0", max_iter=500, zoom=1e20)
    assert a is not b


def test_cache_extended_flag_separate():
    a = compute_reference_orbit("0.0", "1.0", max_iter=400, zoom=1e20)
    b = compute_reference_orbit("0.0", "1.0", max_iter=400, zoom=1e20, extended=True)
    assert a is not b
    assert not a.has_extended and b.has_extended


def test_cache_disabled():
    a = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=500,
                                zoom=1e20, use_cache=False)
    b = compute_reference_orbit(SEAHORSE_RE, SEAHORSE_IM, max_iter=500,
                                zoom=1e20, use_cache=False)
    assert a is not b
    np.testing.assert_array_equal(a.z_re, b.z_re)
