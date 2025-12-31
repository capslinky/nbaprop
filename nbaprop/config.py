"""Configuration for the rebuild."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List
import json
import os


_DEFAULT_PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_points_rebounds_assists",
    "player_threes",
]
_DEFAULT_BOOKMAKERS: List[str] = ["fanduel"]
_DEFAULT_ODDS_CONFIDENCE_MIN = -120
_DEFAULT_ODDS_CONFIDENCE_MAX = 120
_DEFAULT_CONFIDENCE_CAP = 0.65
_DEFAULT_PROP_THRESHOLDS_PATH = "data/prop_thresholds.json"
_DEFAULT_EXCLUDED_PROP_TYPES = ["steals", "blocks", "turnovers"]
_DEFAULT_TOP_PICKS_PER_GAME = 10
_DEFAULT_PICK_RANKING_MODE = "edge_confidence"
_DEFAULT_INJURY_RISK_EDGE_MULTIPLIER = 0.85
_DEFAULT_INJURY_RISK_CONFIDENCE_MULTIPLIER = 0.85
_DEFAULT_PERPLEXITY_ENABLED = False

# Adjustment factor defaults
_DEFAULT_HOME_BOOST = 1.025
_DEFAULT_AWAY_PENALTY = 0.975
_DEFAULT_B2B_PENALTY = 0.92
_DEFAULT_REST_DAY_FACTORS = {
    0: 0.88,   # Same day (edge case)
    1: 0.92,   # Back-to-back (-8%)
    2: 1.00,   # Normal rest (neutral)
    3: 1.02,   # Well rested (+2%)
    4: 1.03,   # Very rested (+3%)
    5: 1.02,   # 4+ days may have some rust
    6: 1.01,   # Extended rest - slight rust factor
}

# Usage rate & shot volume defaults
_DEFAULT_LEAGUE_AVG_USG = 20.0
_DEFAULT_USAGE_FACTOR_WEIGHT = 0.3
_DEFAULT_MAX_USAGE_ADJUSTMENT = 0.05
_DEFAULT_MAX_SHOT_VOLUME_ADJUSTMENT = 0.08
_DEFAULT_SHOT_VOLUME_UP_THRESHOLD = 1.03
_DEFAULT_SHOT_VOLUME_DOWN_THRESHOLD = 0.97

# True shooting efficiency defaults
_DEFAULT_TS_REGRESSION_WEIGHT = 0.3
_DEFAULT_TS_DEVIATION_THRESHOLD = 0.05
_DEFAULT_MAX_TS_ADJUSTMENT = 0.05

# Maximum adjustment caps
_DEFAULT_MAX_VS_TEAM_ADJUSTMENT = 0.10
_DEFAULT_MAX_MINUTES_ADJUSTMENT = 0.10
_DEFAULT_MAX_INJURY_BOOST = 0.15

# Blowout risk defaults
_DEFAULT_BLOWOUT_HIGH_PENALTY = 0.95
_DEFAULT_BLOWOUT_MEDIUM_PENALTY = 0.98

# Defense ranking thresholds
_DEFAULT_SMASH_RANK = 5
_DEFAULT_GOOD_RANK = 10
_DEFAULT_HARD_RANK = 21
_DEFAULT_TOUGH_RANK = 26

# Game total & pace defaults
_DEFAULT_LEAGUE_AVG_TOTAL = 225.0
_DEFAULT_TOTAL_WEIGHT = 0.3
_DEFAULT_HIGH_TOTAL_THRESHOLD = 235.0
_DEFAULT_LOW_TOTAL_THRESHOLD = 215.0
_DEFAULT_FAST_PACE_THRESHOLD = 1.03
_DEFAULT_SLOW_PACE_THRESHOLD = 0.97

# Projection weighting defaults
_DEFAULT_PROJECTION_BIAS = 1.03
_DEFAULT_RECENT_WEIGHT = 0.60
_DEFAULT_OLDER_WEIGHT = 0.40
_DEFAULT_TREND_MULTIPLIER = 0.10

# Trend thresholds
_DEFAULT_TREND_HOT_THRESHOLD = 0.05
_DEFAULT_TREND_COLD_THRESHOLD = -0.05

# Confidence bounds
_DEFAULT_MAX_CONFIDENCE = 0.95
_DEFAULT_MIN_CONFIDENCE_FLOOR = 0.20

# Sample size minimums per prop type
_DEFAULT_MIN_SAMPLE_POINTS = 10
_DEFAULT_MIN_SAMPLE_REBOUNDS = 12
_DEFAULT_MIN_SAMPLE_ASSISTS = 15
_DEFAULT_MIN_SAMPLE_PRA = 10
_DEFAULT_MIN_SAMPLE_THREES = 15
_DEFAULT_MIN_SAMPLE_BLOCKS = 20
_DEFAULT_MIN_SAMPLE_STEALS = 20
_DEFAULT_MIN_SAMPLE_DEFAULT = 15

# Valid prop types
_DEFAULT_VALID_PROP_TYPES = [
    "points", "rebounds", "assists", "pra", "threes",
    "steals", "blocks", "turnovers", "fg3m"
]

# Cache TTLs (seconds)
_DEFAULT_DEFENSE_CACHE_TTL = 3600
_DEFAULT_PACE_CACHE_TTL = 3600
_DEFAULT_INJURY_CACHE_TTL = 1800
_DEFAULT_GAME_LOG_CACHE_TTL = 3600

_DEFAULT_PROP_TYPE_MIN_EDGE = {
    "points": 0.03,
    "rebounds": 0.03,
    "assists": 0.03,
    "pra": 0.03,
    "threes": 0.03,
}
_DEFAULT_PROP_TYPE_MIN_CONFIDENCE = {
    "points": 0.40,
    "rebounds": 0.40,
    "assists": 0.40,
    "pra": 0.40,
    "threes": 0.40,
}


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _coerce_float(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Optional[str], default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Optional[str], default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _coerce_list(value: Optional[str], default: List[str]) -> List[str]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default)


def _parse_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = _strip_quotes(value.strip())
    return data


def _load_config_data(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in payload.items()}
    return _parse_env_file(path)


def _load_thresholds(
    path: str,
    default_edge: Dict[str, float],
    default_conf: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    edge = dict(default_edge)
    conf = dict(default_conf)
    if not path:
        return {"min_edge": edge, "min_confidence": conf}
    thresholds_path = Path(path)
    if not thresholds_path.is_absolute():
        thresholds_path = Path(__file__).resolve().parents[1] / thresholds_path
    if not thresholds_path.exists():
        return {"min_edge": edge, "min_confidence": conf}
    try:
        payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"min_edge": edge, "min_confidence": conf}
    if isinstance(payload, dict):
        min_edge = payload.get("min_edge")
        min_conf = payload.get("min_confidence")
        if isinstance(min_edge, dict):
            edge.update({str(k): float(v) for k, v in min_edge.items() if v is not None})
        if isinstance(min_conf, dict):
            conf.update({str(k): float(v) for k, v in min_conf.items() if v is not None})
    return {"min_edge": edge, "min_confidence": conf}


@dataclass
class Config:
    # API and odds settings
    odds_api_key: str
    odds_max_events: int
    odds_max_players: int
    odds_prop_markets: List[str]
    odds_bookmakers: List[str]
    odds_min: Optional[int]
    odds_max: Optional[int]
    odds_confidence_min: Optional[int]
    odds_confidence_max: Optional[int]
    confidence_cap_outside_band: float
    min_edge_threshold: float
    min_confidence: float
    nba_api_delay: float
    cache_dir: str
    perplexity_api_key: str
    market_blend: float
    max_picks: int
    historical_props_path: str
    run_date: str
    auto_review: bool
    review_step: float
    review_update_env: bool
    calibration_path: str
    calibration_min_samples: int
    clv_enabled: bool
    clv_snapshot_ttl: int
    prop_thresholds_path: str
    prop_type_min_edge: Dict[str, float]
    prop_type_min_confidence: Dict[str, float]
    excluded_prop_types: List[str]
    top_picks_per_game: int
    pick_ranking_mode: str
    injury_risk_edge_multiplier: float
    injury_risk_confidence_multiplier: float
    perplexity_enabled: bool

    # Adjustment factors
    home_boost: float = _DEFAULT_HOME_BOOST
    away_penalty: float = _DEFAULT_AWAY_PENALTY
    b2b_penalty: float = _DEFAULT_B2B_PENALTY
    rest_day_factors: Dict[int, float] = None  # type: ignore

    # Usage rate & shot volume
    league_avg_usg: float = _DEFAULT_LEAGUE_AVG_USG
    usage_factor_weight: float = _DEFAULT_USAGE_FACTOR_WEIGHT
    max_usage_adjustment: float = _DEFAULT_MAX_USAGE_ADJUSTMENT
    max_shot_volume_adjustment: float = _DEFAULT_MAX_SHOT_VOLUME_ADJUSTMENT
    shot_volume_up_threshold: float = _DEFAULT_SHOT_VOLUME_UP_THRESHOLD
    shot_volume_down_threshold: float = _DEFAULT_SHOT_VOLUME_DOWN_THRESHOLD

    # True shooting efficiency
    ts_regression_weight: float = _DEFAULT_TS_REGRESSION_WEIGHT
    ts_deviation_threshold: float = _DEFAULT_TS_DEVIATION_THRESHOLD
    max_ts_adjustment: float = _DEFAULT_MAX_TS_ADJUSTMENT

    # Maximum adjustment caps
    max_vs_team_adjustment: float = _DEFAULT_MAX_VS_TEAM_ADJUSTMENT
    max_minutes_adjustment: float = _DEFAULT_MAX_MINUTES_ADJUSTMENT
    max_injury_boost: float = _DEFAULT_MAX_INJURY_BOOST

    # Blowout risk
    blowout_high_penalty: float = _DEFAULT_BLOWOUT_HIGH_PENALTY
    blowout_medium_penalty: float = _DEFAULT_BLOWOUT_MEDIUM_PENALTY

    # Defense ranking thresholds
    smash_rank: int = _DEFAULT_SMASH_RANK
    good_rank: int = _DEFAULT_GOOD_RANK
    hard_rank: int = _DEFAULT_HARD_RANK
    tough_rank: int = _DEFAULT_TOUGH_RANK

    # Game total & pace
    league_avg_total: float = _DEFAULT_LEAGUE_AVG_TOTAL
    total_weight: float = _DEFAULT_TOTAL_WEIGHT
    high_total_threshold: float = _DEFAULT_HIGH_TOTAL_THRESHOLD
    low_total_threshold: float = _DEFAULT_LOW_TOTAL_THRESHOLD
    fast_pace_threshold: float = _DEFAULT_FAST_PACE_THRESHOLD
    slow_pace_threshold: float = _DEFAULT_SLOW_PACE_THRESHOLD

    # Projection weighting
    projection_bias: float = _DEFAULT_PROJECTION_BIAS
    recent_weight: float = _DEFAULT_RECENT_WEIGHT
    older_weight: float = _DEFAULT_OLDER_WEIGHT
    trend_multiplier: float = _DEFAULT_TREND_MULTIPLIER

    # Trend thresholds
    trend_hot_threshold: float = _DEFAULT_TREND_HOT_THRESHOLD
    trend_cold_threshold: float = _DEFAULT_TREND_COLD_THRESHOLD

    # Confidence bounds
    max_confidence: float = _DEFAULT_MAX_CONFIDENCE
    min_confidence_floor: float = _DEFAULT_MIN_CONFIDENCE_FLOOR

    # Sample size minimums per prop type
    min_sample_points: int = _DEFAULT_MIN_SAMPLE_POINTS
    min_sample_rebounds: int = _DEFAULT_MIN_SAMPLE_REBOUNDS
    min_sample_assists: int = _DEFAULT_MIN_SAMPLE_ASSISTS
    min_sample_pra: int = _DEFAULT_MIN_SAMPLE_PRA
    min_sample_threes: int = _DEFAULT_MIN_SAMPLE_THREES
    min_sample_blocks: int = _DEFAULT_MIN_SAMPLE_BLOCKS
    min_sample_steals: int = _DEFAULT_MIN_SAMPLE_STEALS
    min_sample_default: int = _DEFAULT_MIN_SAMPLE_DEFAULT

    # Valid prop types
    valid_prop_types: List[str] = None  # type: ignore

    # Cache TTLs
    defense_cache_ttl: int = _DEFAULT_DEFENSE_CACHE_TTL
    pace_cache_ttl: int = _DEFAULT_PACE_CACHE_TTL
    injury_cache_ttl: int = _DEFAULT_INJURY_CACHE_TTL
    game_log_cache_ttl: int = _DEFAULT_GAME_LOG_CACHE_TTL

    def __post_init__(self) -> None:
        if self.rest_day_factors is None:
            self.rest_day_factors = dict(_DEFAULT_REST_DAY_FACTORS)
        if self.valid_prop_types is None:
            self.valid_prop_types = list(_DEFAULT_VALID_PROP_TYPES)

    def get_min_sample_size(self, prop_type: str) -> int:
        """Get minimum sample size for a prop type."""
        sample_map = {
            "points": self.min_sample_points,
            "rebounds": self.min_sample_rebounds,
            "assists": self.min_sample_assists,
            "pra": self.min_sample_pra,
            "threes": self.min_sample_threes,
            "fg3m": self.min_sample_threes,
            "blocks": self.min_sample_blocks,
            "steals": self.min_sample_steals,
        }
        return sample_map.get(prop_type.lower(), self.min_sample_default)

    @classmethod
    def from_env(cls) -> "Config":
        thresholds_path = os.environ.get("PROP_THRESHOLDS_PATH", _DEFAULT_PROP_THRESHOLDS_PATH)
        thresholds = _load_thresholds(
            thresholds_path,
            _DEFAULT_PROP_TYPE_MIN_EDGE,
            _DEFAULT_PROP_TYPE_MIN_CONFIDENCE,
        )
        return cls(
            odds_api_key=os.environ.get("ODDS_API_KEY", ""),
            odds_max_events=_coerce_int(os.environ.get("ODDS_MAX_EVENTS"), 5),
            odds_max_players=_coerce_int(os.environ.get("ODDS_MAX_PLAYERS"), 0),
            odds_prop_markets=_coerce_list(os.environ.get("ODDS_PROP_MARKETS"), _DEFAULT_PROP_MARKETS),
            odds_bookmakers=_coerce_list(os.environ.get("ODDS_BOOKMAKERS"), _DEFAULT_BOOKMAKERS),
            odds_min=_coerce_optional_int(os.environ.get("ODDS_MIN")),
            odds_max=_coerce_optional_int(os.environ.get("ODDS_MAX")),
            odds_confidence_min=(
                _coerce_optional_int(os.environ.get("ODDS_CONFIDENCE_MIN"))
                if os.environ.get("ODDS_CONFIDENCE_MIN") not in (None, "")
                else _DEFAULT_ODDS_CONFIDENCE_MIN
            ),
            odds_confidence_max=(
                _coerce_optional_int(os.environ.get("ODDS_CONFIDENCE_MAX"))
                if os.environ.get("ODDS_CONFIDENCE_MAX") not in (None, "")
                else _DEFAULT_ODDS_CONFIDENCE_MAX
            ),
            confidence_cap_outside_band=_coerce_float(
                os.environ.get("CONFIDENCE_CAP_OUTSIDE_BAND"),
                _DEFAULT_CONFIDENCE_CAP,
            ),
            min_edge_threshold=_coerce_float(os.environ.get("MIN_EDGE_THRESHOLD"), 0.03),
            min_confidence=_coerce_float(os.environ.get("MIN_CONFIDENCE"), 0.4),
            nba_api_delay=_coerce_float(os.environ.get("NBA_API_DELAY"), 1.5),
            cache_dir=os.environ.get("NBAPROP_CACHE_DIR", ".cache"),
            perplexity_api_key=os.environ.get("PERPLEXITY_API_KEY", ""),
            market_blend=_coerce_float(os.environ.get("MARKET_BLEND"), 0.7),
            max_picks=_coerce_int(os.environ.get("MAX_PICKS"), 50),
            historical_props_path=os.environ.get("HISTORICAL_PROPS_PATH", ""),
            run_date=os.environ.get("RUN_DATE", ""),
            auto_review=_coerce_bool(os.environ.get("AUTO_REVIEW"), False),
            review_step=_coerce_float(os.environ.get("REVIEW_STEP"), 0.1),
            review_update_env=_coerce_bool(os.environ.get("REVIEW_UPDATE_ENV"), True),
            calibration_path=os.environ.get("NBAPROP_CALIBRATION_PATH", ""),
            calibration_min_samples=_coerce_int(os.environ.get("CALIBRATION_MIN_SAMPLES"), 30),
            clv_enabled=_coerce_bool(os.environ.get("CLV_ENABLED"), False),
            clv_snapshot_ttl=_coerce_int(os.environ.get("CLV_SNAPSHOT_TTL"), 600),
            prop_thresholds_path=thresholds_path,
            prop_type_min_edge=thresholds["min_edge"],
            prop_type_min_confidence=thresholds["min_confidence"],
            excluded_prop_types=_coerce_list(
                os.environ.get("EXCLUDED_PROP_TYPES"),
                _DEFAULT_EXCLUDED_PROP_TYPES,
            ),
            top_picks_per_game=_coerce_int(
                os.environ.get("TOP_PICKS_PER_GAME"),
                _DEFAULT_TOP_PICKS_PER_GAME,
            ),
            pick_ranking_mode=os.environ.get("PICK_RANKING_MODE", _DEFAULT_PICK_RANKING_MODE),
            injury_risk_edge_multiplier=_coerce_float(
                os.environ.get("INJURY_RISK_EDGE_MULTIPLIER"),
                _DEFAULT_INJURY_RISK_EDGE_MULTIPLIER,
            ),
            injury_risk_confidence_multiplier=_coerce_float(
                os.environ.get("INJURY_RISK_CONFIDENCE_MULTIPLIER"),
                _DEFAULT_INJURY_RISK_CONFIDENCE_MULTIPLIER,
            ),
            perplexity_enabled=_coerce_bool(
                os.environ.get("PERPLEXITY_ENABLED"),
                _DEFAULT_PERPLEXITY_ENABLED,
            ),
        )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        env_config = cls.from_env()
        if not config_path:
            return env_config

        file_data = _load_config_data(Path(config_path))
        thresholds_path = file_data.get("PROP_THRESHOLDS_PATH", env_config.prop_thresholds_path)
        thresholds = _load_thresholds(
            thresholds_path,
            env_config.prop_type_min_edge,
            env_config.prop_type_min_confidence,
        )
        return cls(
            odds_api_key=file_data.get("ODDS_API_KEY", env_config.odds_api_key),
            odds_max_events=_coerce_int(
                file_data.get("ODDS_MAX_EVENTS"),
                env_config.odds_max_events,
            ),
            odds_max_players=_coerce_int(
                file_data.get("ODDS_MAX_PLAYERS"),
                env_config.odds_max_players,
            ),
            odds_prop_markets=_coerce_list(
                file_data.get("ODDS_PROP_MARKETS"),
                env_config.odds_prop_markets,
            ),
            odds_bookmakers=_coerce_list(
                file_data.get("ODDS_BOOKMAKERS"),
                env_config.odds_bookmakers,
            ),
            odds_min=(
                _coerce_optional_int(file_data.get("ODDS_MIN"))
                if file_data.get("ODDS_MIN") not in (None, "")
                else env_config.odds_min
            ),
            odds_max=(
                _coerce_optional_int(file_data.get("ODDS_MAX"))
                if file_data.get("ODDS_MAX") not in (None, "")
                else env_config.odds_max
            ),
            odds_confidence_min=(
                _coerce_optional_int(file_data.get("ODDS_CONFIDENCE_MIN"))
                if file_data.get("ODDS_CONFIDENCE_MIN") not in (None, "")
                else env_config.odds_confidence_min
            ),
            odds_confidence_max=(
                _coerce_optional_int(file_data.get("ODDS_CONFIDENCE_MAX"))
                if file_data.get("ODDS_CONFIDENCE_MAX") not in (None, "")
                else env_config.odds_confidence_max
            ),
            confidence_cap_outside_band=_coerce_float(
                file_data.get("CONFIDENCE_CAP_OUTSIDE_BAND"),
                env_config.confidence_cap_outside_band,
            ),
            min_edge_threshold=_coerce_float(
                file_data.get("MIN_EDGE_THRESHOLD"),
                env_config.min_edge_threshold,
            ),
            min_confidence=_coerce_float(
                file_data.get("MIN_CONFIDENCE"),
                env_config.min_confidence,
            ),
            nba_api_delay=_coerce_float(
                file_data.get("NBA_API_DELAY"),
                env_config.nba_api_delay,
            ),
            cache_dir=file_data.get("NBAPROP_CACHE_DIR", env_config.cache_dir),
            perplexity_api_key=file_data.get("PERPLEXITY_API_KEY", env_config.perplexity_api_key),
            market_blend=_coerce_float(
                file_data.get("MARKET_BLEND"),
                env_config.market_blend,
            ),
            max_picks=_coerce_int(
                file_data.get("MAX_PICKS"),
                env_config.max_picks,
            ),
            historical_props_path=file_data.get(
                "HISTORICAL_PROPS_PATH",
                env_config.historical_props_path,
            ),
            run_date=file_data.get("RUN_DATE", env_config.run_date),
            auto_review=_coerce_bool(
                file_data.get("AUTO_REVIEW"),
                env_config.auto_review,
            ),
            review_step=_coerce_float(
                file_data.get("REVIEW_STEP"),
                env_config.review_step,
            ),
            review_update_env=_coerce_bool(
                file_data.get("REVIEW_UPDATE_ENV"),
                env_config.review_update_env,
            ),
            calibration_path=file_data.get("NBAPROP_CALIBRATION_PATH", env_config.calibration_path),
            calibration_min_samples=_coerce_int(
                file_data.get("CALIBRATION_MIN_SAMPLES"),
                env_config.calibration_min_samples,
            ),
            clv_enabled=_coerce_bool(
                file_data.get("CLV_ENABLED"),
                env_config.clv_enabled,
            ),
            clv_snapshot_ttl=_coerce_int(
                file_data.get("CLV_SNAPSHOT_TTL"),
                env_config.clv_snapshot_ttl,
            ),
            prop_thresholds_path=thresholds_path,
            prop_type_min_edge=thresholds["min_edge"],
            prop_type_min_confidence=thresholds["min_confidence"],
            excluded_prop_types=_coerce_list(
                file_data.get("EXCLUDED_PROP_TYPES"),
                env_config.excluded_prop_types,
            ),
            top_picks_per_game=_coerce_int(
                file_data.get("TOP_PICKS_PER_GAME"),
                env_config.top_picks_per_game,
            ),
            pick_ranking_mode=file_data.get("PICK_RANKING_MODE", env_config.pick_ranking_mode),
            injury_risk_edge_multiplier=_coerce_float(
                file_data.get("INJURY_RISK_EDGE_MULTIPLIER"),
                env_config.injury_risk_edge_multiplier,
            ),
            injury_risk_confidence_multiplier=_coerce_float(
                file_data.get("INJURY_RISK_CONFIDENCE_MULTIPLIER"),
                env_config.injury_risk_confidence_multiplier,
            ),
            perplexity_enabled=_coerce_bool(
                file_data.get("PERPLEXITY_ENABLED"),
                env_config.perplexity_enabled,
            ),
        )

    def to_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    def prop_thresholds_for(self, prop_type: str) -> Dict[str, float]:
        key = (prop_type or "").lower()
        return {
            "min_edge": self.prop_type_min_edge.get(key, self.min_edge_threshold),
            "min_confidence": self.prop_type_min_confidence.get(key, self.min_confidence),
        }
