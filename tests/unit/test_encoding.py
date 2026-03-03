"""Unit tests for temporal encoding and geohash utilities."""

import numpy as np
import pytest

from swaadstack.utils.encoding import encode_temporal_features, get_mealtime_label, geohash_to_bucket


class TestTemporalEncoding:
    def test_encoding_shape(self):
        assert encode_temporal_features(12, 3).shape == (4,)

    def test_encoding_range(self):
        for hour in range(24):
            for dow in range(7):
                result = encode_temporal_features(hour, dow)
                assert np.all(result >= -1.0)
                assert np.all(result <= 1.0)

    def test_cyclical_continuity(self):
        enc_23 = encode_temporal_features(23, 0)
        enc_0 = encode_temporal_features(0, 0)
        enc_12 = encode_temporal_features(12, 0)
        assert np.linalg.norm(enc_23 - enc_0) < np.linalg.norm(enc_23 - enc_12)

    def test_mealtime_labels(self):
        assert get_mealtime_label(8) == "breakfast"
        assert get_mealtime_label(13) == "lunch"
        assert get_mealtime_label(16) == "snacks"
        assert get_mealtime_label(20) == "dinner"
        assert get_mealtime_label(2) == "late_night"


class TestGeohash:
    def test_geohash_bucket_consistency(self):
        assert geohash_to_bucket("tdr1y") == geohash_to_bucket("tdr1y")

    def test_geohash_bucket_range(self):
        for geohash in ["tdr1y", "tdr1x", "tdnu8", "abc123"]:
            bucket = geohash_to_bucket(geohash, num_buckets=100)
            assert 0 <= bucket < 100
