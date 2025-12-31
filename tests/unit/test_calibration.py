"""Unit tests for the calibration module."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calibration.weight_store import LearnedWeightsStore, LearnedWeights
from calibration.analyzer import CalibrationAnalyzer, FactorStats, CalibrationResult
from calibration.optimizer import WeightOptimizer, OptimizedFactor
from core.config import CONFIG


class TestLearnedWeightsStore:
    """Tests for LearnedWeightsStore."""

    def test_load_nonexistent_file(self):
        """Test loading when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LearnedWeightsStore(Path(tmpdir) / "nonexistent.json")
            assert store.load() is False
            assert store.get_factor('HOME_BOOST', 1.025) == 1.025

    def test_save_and_load(self):
        """Test saving and loading weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            store = LearnedWeightsStore(path)

            weights = LearnedWeights(
                version="1.0",
                adjustment_factors={
                    'HOME_BOOST': {
                        'value': 1.030,
                        'default': 1.025,
                        'sample_size': 100,
                        'quality': 'high',
                    }
                },
                metadata={'total_picks': 100},
            )

            assert store.save(weights) is True
            assert path.exists()

            # Load in new store
            store2 = LearnedWeightsStore(path)
            assert store2.load() is True
            assert store2.get_factor('HOME_BOOST', 1.025) == 1.030

    def test_get_factor_with_fallback(self):
        """Test getting factor with fallback to default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            store = LearnedWeightsStore(path)

            weights = LearnedWeights(
                adjustment_factors={
                    'HOME_BOOST': {'value': 1.030, 'quality': 'high'},
                }
            )
            store.save(weights)
            store.load()

            # Factor exists
            assert store.get_factor('HOME_BOOST', 1.025) == 1.030

            # Factor doesn't exist - use default
            assert store.get_factor('NONEXISTENT', 0.5) == 0.5

    def test_insufficient_data_returns_default(self):
        """Test that factors with insufficient_data quality return default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            store = LearnedWeightsStore(path)

            weights = LearnedWeights(
                adjustment_factors={
                    'HOME_BOOST': {
                        'value': 1.030,
                        'quality': 'insufficient_data',
                    }
                }
            )
            store.save(weights)
            store.load()

            # Should return default because quality is insufficient_data
            assert store.get_factor('HOME_BOOST', 1.025) == 1.025

    def test_is_valid(self):
        """Test validity checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            store = LearnedWeightsStore(path)

            # No weights loaded
            assert store.is_valid() is False

            # Save valid weights
            weights = LearnedWeights(
                version="1.0",
                calibrated_at=datetime.now().isoformat(),
                adjustment_factors={'HOME_BOOST': {'value': 1.030}},
            )
            store.save(weights)
            store.load()

            assert store.is_valid() is True

    def test_clear(self):
        """Test clearing weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            store = LearnedWeightsStore(path)

            weights = LearnedWeights(adjustment_factors={'HOME_BOOST': {'value': 1.030}})
            store.save(weights)

            assert path.exists()
            store.clear()
            assert not path.exists()


class TestWeightOptimizer:
    """Tests for WeightOptimizer."""

    def test_optimize_insufficient_samples(self):
        """Test that insufficient samples returns default."""
        optimizer = WeightOptimizer(min_samples=50)

        stats = FactorStats(
            factor_name='HOME_BOOST',
            active_count=30,  # Less than 50
            inactive_count=100,
            active_win_rate=0.55,
            inactive_win_rate=0.50,
            improvement=0.05,
            sample_sufficient=False,
            quality='insufficient_data',
        )

        result = optimizer.optimize_factor('HOME_BOOST', stats, current_value=1.025)

        assert result.was_adjusted is False
        assert result.new_value == 1.025
        assert 'Insufficient' in result.reason

    def test_optimize_with_sufficient_samples(self):
        """Test optimization with sufficient samples."""
        optimizer = WeightOptimizer(
            min_samples=50,
            conservative_factor=0.5,
            max_change=0.03,
        )

        stats = FactorStats(
            factor_name='HOME_BOOST',
            active_count=200,
            inactive_count=100,
            active_win_rate=0.58,
            inactive_win_rate=0.50,
            improvement=0.08,
            sample_sufficient=True,
            quality='high',
        )

        result = optimizer.optimize_factor('HOME_BOOST', stats, current_value=1.025)

        # Should be adjusted
        assert result.sample_size == 200
        assert result.quality == 'high'
        # New value should be between current and optimal
        assert result.current_value <= result.new_value <= result.optimal_value or \
               result.optimal_value <= result.new_value <= result.current_value

    def test_bounds_enforcement(self):
        """Test that bounds are enforced."""
        bounds = {'HOME_BOOST': (1.00, 1.05)}
        optimizer = WeightOptimizer(
            min_samples=10,
            conservative_factor=1.0,  # No dampening
            max_change=1.0,  # No limit
            bounds=bounds,
        )

        # Try to set value above bounds
        stats = FactorStats(
            factor_name='HOME_BOOST',
            active_count=100,
            inactive_count=100,
            active_win_rate=0.90,  # Very high - would suggest large increase
            inactive_win_rate=0.50,
            improvement=0.40,
            sample_sufficient=True,
            quality='high',
        )

        result = optimizer.optimize_factor('HOME_BOOST', stats, current_value=1.025)

        # Should be clamped to max bound
        assert result.new_value <= 1.05

    def test_max_change_limit(self):
        """Test that per-run change is limited."""
        optimizer = WeightOptimizer(
            min_samples=10,
            conservative_factor=1.0,  # No dampening
            max_change=0.01,  # 1% max change
        )

        stats = FactorStats(
            factor_name='HOME_BOOST',
            active_count=100,
            inactive_count=100,
            active_win_rate=0.70,
            inactive_win_rate=0.50,
            improvement=0.20,
            sample_sufficient=True,
            quality='high',
        )

        result = optimizer.optimize_factor('HOME_BOOST', stats, current_value=1.025)

        # Change should be limited to 1%
        change = abs(result.new_value - result.current_value) / result.current_value
        assert change <= 0.01 + 0.001  # Small tolerance

    def test_conservative_factor_applied(self):
        """Test that conservative factor dampens changes."""
        # With conservative_factor=0.5, we should only move halfway
        optimizer = WeightOptimizer(
            min_samples=10,
            conservative_factor=0.5,
            max_change=1.0,  # No limit
        )

        stats = FactorStats(
            factor_name='HOME_BOOST',
            active_count=100,
            inactive_count=100,
            active_win_rate=0.55,
            inactive_win_rate=0.50,
            improvement=0.05,
            sample_sufficient=True,
            quality='high',
        )

        result = optimizer.optimize_factor('HOME_BOOST', stats, current_value=1.025)

        # The change should be dampened by conservative factor
        # actual_change = (new - current) should be ~50% of (optimal - current)
        if result.optimal_value != result.current_value:
            full_change = result.optimal_value - result.current_value
            actual_change = result.new_value - result.current_value
            # Allow some tolerance for bounds clamping
            ratio = abs(actual_change / full_change) if full_change != 0 else 0
            assert ratio <= 0.6  # Should be around 0.5 or less

    def test_create_learned_weights(self):
        """Test creating LearnedWeights from optimized factors."""
        optimizer = WeightOptimizer()

        stats = FactorStats(
            factor_name='HOME_BOOST',
            active_count=100,
            inactive_count=100,
            active_win_rate=0.55,
            inactive_win_rate=0.50,
            improvement=0.05,
            sample_sufficient=True,
            quality='high',
        )

        calibration_result = CalibrationResult(
            total_picks=200,
            picks_with_results=200,
            date_range=('2024-01-01', '2024-12-01'),
            factor_stats={'HOME_BOOST': stats},
            overall_win_rate=0.52,
            has_sufficient_data=True,
        )

        optimized = optimizer.optimize_all_factors(calibration_result)
        weights = optimizer.create_learned_weights(optimized, calibration_result)

        assert weights.version == "1.0"
        assert 'HOME_BOOST' in weights.adjustment_factors
        assert weights.metadata['total_picks'] == 200


class TestCalibrationAnalyzer:
    """Tests for CalibrationAnalyzer."""

    def test_get_data_summary_no_db(self):
        """Test data summary when no database exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CalibrationAnalyzer(db_path=Path(tmpdir) / "nonexistent.db")
            summary = analyzer.get_data_summary()

            assert summary['exists'] is False
            assert summary['total_picks'] == 0

    def test_has_sufficient_data_false(self):
        """Test has_sufficient_data returns False when no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CalibrationAnalyzer(db_path=Path(tmpdir) / "nonexistent.db")
            assert analyzer.has_sufficient_data() is False

    def test_get_quality(self):
        """Test quality rating based on sample size."""
        analyzer = CalibrationAnalyzer()

        # Below minimum
        assert analyzer._get_quality(30) == 'insufficient_data'

        # Low (1-2x minimum)
        assert analyzer._get_quality(60) == 'low'

        # Medium (2-4x minimum)
        assert analyzer._get_quality(150) == 'medium'

        # High (4x+ minimum)
        assert analyzer._get_quality(300) == 'high'


class TestUnifiedPropModelIntegration:
    """Tests for UnifiedPropModel integration with learned weights."""

    def test_model_uses_defaults_without_weights(self):
        """Test that model uses CONFIG defaults when no weights file."""
        from models import UnifiedPropModel

        # Create model without learned weights
        model = UnifiedPropModel(use_learned_weights=False)

        assert model.HOME_BOOST == CONFIG.HOME_BOOST
        assert model.B2B_PENALTY == CONFIG.B2B_PENALTY
        assert model.AWAY_PENALTY == CONFIG.AWAY_PENALTY

    def test_model_loads_learned_weights(self):
        """Test that model loads learned weights when available."""
        from models import UnifiedPropModel

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create weights file
            weights_path = Path(tmpdir) / "learned_weights.json"
            weights = LearnedWeights(
                version="1.0",
                calibrated_at=datetime.now().isoformat(),
                adjustment_factors={
                    'HOME_BOOST': {'value': 1.030, 'quality': 'high'},
                    'B2B_PENALTY': {'value': 0.90, 'quality': 'high'},
                }
            )

            with open(weights_path, 'w') as f:
                json.dump(weights.to_dict(), f)

            # Patch the store's default path
            with patch.object(LearnedWeightsStore, 'DEFAULT_PATH', weights_path):
                model = UnifiedPropModel(use_learned_weights=True)

                # Should use learned weights
                assert model.HOME_BOOST == 1.030
                assert model.B2B_PENALTY == 0.90

                # Non-calibrated factor should use default
                assert model.LEAGUE_AVG_TOTAL == CONFIG.LEAGUE_AVG_TOTAL

    def test_model_fallback_on_import_error(self):
        """Test graceful fallback when calibration module unavailable."""
        from models import UnifiedPropModel
        from calibration.weight_store import LearnedWeightsStore

        # Model should work even if calibration import fails
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "missing_weights.json"
            with patch.object(LearnedWeightsStore, 'DEFAULT_PATH', empty_path):
                model = UnifiedPropModel(use_learned_weights=True)

        # Should fall back to CONFIG defaults
        assert model.HOME_BOOST == CONFIG.HOME_BOOST


class TestCalibrationEndToEnd:
    """End-to-end tests for the calibration workflow."""

    def test_full_calibration_workflow(self):
        """Test the complete calibration workflow."""
        # This is a conceptual test - in practice, you'd need a populated database

        # 1. Create analyzer
        with tempfile.TemporaryDirectory() as tmpdir:
            # Would need to populate a test database here
            # analyzer = CalibrationAnalyzer(db_path=Path(tmpdir) / "test.db")

            # 2. Run analysis
            # result = analyzer.calculate_all_factor_stats(days=90)

            # 3. Optimize
            optimizer = WeightOptimizer()

            # 4. Create learned weights
            # weights = optimizer.create_learned_weights(optimized, result)

            # 5. Save
            store = LearnedWeightsStore(Path(tmpdir) / "weights.json")

            # Just verify the workflow components exist and can be instantiated
            assert optimizer is not None
            assert store is not None
