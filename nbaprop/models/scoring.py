"""Odds-aware scoring helpers."""

from typing import Dict, Optional
import math

from nbaprop.models.calibration import apply_calibration


_OUT_STATUSES = {
    "OUT",
    "O",
    "DNP",
    "INJURED",
    "SUSPENDED",
    "DOUBTFUL",
    "D",
}
_RISK_STATUSES = {
    "QUESTIONABLE",
    "Q",
    "GTD",
    "GAME TIME DECISION",
    "GAME-TIME DECISION",
}
_SOFT_STATUSES = {
    "PROBABLE",
    "P",
}

_PROP_RISK_FACTORS = {
    "points": 1.0,
    "rebounds": 0.95,
    "assists": 1.0,
    "threes": 1.1,
    "pra": 1.05,
    "steals": 1.2,
    "blocks": 1.2,
    "turnovers": 1.15,
}

_USAGE_SENSITIVE_PROPS = {
    "points",
    "assists",
    "pra",
    "threes",
    "rebounds",
}

_HIGH_VARIANCE_PROPS = {
    "steals",
    "blocks",
    "turnovers",
}

_MEDIUM_VARIANCE_PROPS = {
    "threes",
}


def _american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _confidence_tier(confidence: float) -> str:
    if confidence >= 0.7:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"


def _normalize_prop_type(prop_type: Optional[str]) -> str:
    prop = (prop_type or "").lower()
    if prop in ("pts", "points"):
        return "points"
    if prop in ("reb", "rebounds"):
        return "rebounds"
    if prop in ("ast", "assists"):
        return "assists"
    if prop in ("threes", "3pt", "fg3m"):
        return "threes"
    if prop in ("pra", "pts_reb_ast"):
        return "pra"
    if prop in ("stl", "steals"):
        return "steals"
    if prop in ("blk", "blocks"):
        return "blocks"
    if prop in ("tov", "turnovers"):
        return "turnovers"
    return prop or "unknown"


def _risk_factor(prop_type: Optional[str]) -> float:
    return _PROP_RISK_FACTORS.get(_normalize_prop_type(prop_type), 1.0)


def _volatility_factor(std: Optional[float], line: Optional[float]) -> float:
    if std in (None, 0, 0.0) or line in (None, 0, 0.0):
        return 1.0
    try:
        ratio = float(std) / max(1.0, abs(float(line)))
    except (TypeError, ValueError):
        return 1.0
    if ratio >= 0.4:
        return 1.15
    if ratio >= 0.3:
        return 1.08
    if ratio <= 0.15:
        return 0.9
    return 1.0


def _volatility_ratio(std: Optional[float], line: Optional[float]) -> Optional[float]:
    if std in (None, 0, 0.0) or line in (None, 0, 0.0):
        return None
    try:
        return abs(float(std)) / max(1.0, abs(float(line)))
    except (TypeError, ValueError):
        return None


def _confidence_volatility_penalty(prop_type: Optional[str], ratio: Optional[float]) -> float:
    if ratio is None:
        return 1.0
    prop = _normalize_prop_type(prop_type)
    if prop in _HIGH_VARIANCE_PROPS:
        if ratio >= 0.45:
            return 0.72
        if ratio >= 0.35:
            return 0.82
        if ratio >= 0.25:
            return 0.9
        return 1.0
    if prop in _MEDIUM_VARIANCE_PROPS:
        if ratio >= 0.45:
            return 0.8
        if ratio >= 0.35:
            return 0.9
        return 1.0
    if ratio >= 0.5:
        return 0.85
    if ratio >= 0.4:
        return 0.92
    return 1.0


def _adjust_thresholds(min_edge: float, min_confidence: float, prop_type: Optional[str],
                       std: Optional[float], line: Optional[float]) -> Dict[str, float]:
    prop_factor = _risk_factor(prop_type)
    vol_factor = _volatility_factor(std, line)
    edge_factor = _clamp(prop_factor * vol_factor, 0.7, 1.6)
    conf_factor = _clamp(0.9 + ((prop_factor - 1.0) * 0.6) + ((vol_factor - 1.0) * 0.6), 0.75, 1.4)
    return {
        "edge": min_edge * edge_factor,
        "confidence": min_confidence * conf_factor,
        "edge_factor": edge_factor,
        "conf_factor": conf_factor,
    }


def score_prop(
    row: Dict,
    min_edge: float = 0.03,
    min_confidence: float = 0.4,
    prop_type_min_edge: Optional[Dict[str, float]] = None,
    prop_type_min_confidence: Optional[Dict[str, float]] = None,
    excluded_prop_types: Optional[list] = None,
    injury_risk_edge_multiplier: float = 1.0,
    injury_risk_confidence_multiplier: float = 1.0,
    market_blend: float = 0.7,
    calibration: Optional[Dict] = None,
    odds_min: Optional[int] = None,
    odds_max: Optional[int] = None,
    odds_confidence_min: Optional[int] = None,
    odds_confidence_max: Optional[int] = None,
    confidence_cap_outside_band: Optional[float] = 0.65,
) -> Dict:
    """Score a single prop row with probability, edge, and confidence."""
    features = row.get("features", {})
    line = features.get("line")
    prop_type = features.get("prop_type")
    prop_key = _normalize_prop_type(prop_type)
    if prop_type_min_edge and prop_key in prop_type_min_edge:
        min_edge = prop_type_min_edge[prop_key]
    if prop_type_min_confidence and prop_key in prop_type_min_confidence:
        min_confidence = prop_type_min_confidence[prop_key]
    injury_status = (features.get("injury_status") or "").upper()
    injury_source = features.get("injury_source")
    injury_detail = features.get("injury_detail")
    market_blend = _clamp(float(market_blend), 0.0, 1.0)
    base_output = {
        "prop_id": row.get("prop_id"),
        "player_id": features.get("player_id"),
        "player_name": features.get("player_name"),
        "prop_type": prop_type,
        "side": features.get("side"),
        "line": line,
        "event_id": features.get("event_id"),
        "home_team": features.get("home_team"),
        "away_team": features.get("away_team"),
        "game_time": features.get("game_time"),
        "player_team": features.get("player_team"),
        "opponent_team": features.get("opponent_team"),
        "is_home": features.get("is_home"),
        "rest_days": features.get("rest_days"),
        "b2b": features.get("b2b"),
        "odds": features.get("odds"),
        "market_prob": features.get("market_prob"),
        "market_over_odds": features.get("market_over_odds"),
        "market_under_odds": features.get("market_under_odds"),
        "expected_usage_factor": features.get("expected_usage_factor"),
        "expected_usage_notes": features.get("expected_usage_notes"),
        "avg_5": features.get("recent_avg"),
        "avg_10": features.get("avg_10"),
        "avg_15": features.get("avg_15"),
        "trend": features.get("trend"),
        "recent_std": features.get("recent_std"),
        "recent_hit_rate_10": None,
        "recent_hit_rate_15": None,
        "recent_streak": None,
        "team_out_count": features.get("team_out_count"),
        "team_questionable_count": features.get("team_questionable_count"),
        "team_probable_count": features.get("team_probable_count"),
        "team_out_detail": features.get("team_out_detail"),
        "team_questionable_detail": features.get("team_questionable_detail"),
        "team_probable_detail": features.get("team_probable_detail"),
        "injury_status": injury_status or None,
        "injury_source": injury_source,
        "injury_detail": injury_detail,
        "edge_threshold": None,
        "confidence_threshold": None,
        "risk_factor": None,
        "volatility_factor": None,
        "market_vig": None,
        "calibration_source": None,
        "calibration_slope": None,
        "calibration_intercept": None,
        "odds_min": odds_min,
        "odds_max": odds_max,
        "volatility_ratio": None,
        "odds_confidence_min": odds_confidence_min,
        "odds_confidence_max": odds_confidence_max,
        "confidence_cap_outside_band": confidence_cap_outside_band,
        "reason": None,
    }

    excluded = {str(item).lower() for item in (excluded_prop_types or [])}
    if prop_key in excluded:
        base_output.update({
            "edge": 0.0,
            "confidence": 0.0,
            "pick": "PASS",
            "reason": "EXCLUDED_PROP_TYPE",
        })
        return base_output

    if injury_status in _OUT_STATUSES:
        base_output.update({
            "edge": 0.0,
            "confidence": 0.0,
            "pick": "PASS",
            "reason": f"INJURY_{injury_status}",
        })
        return base_output
    if line in (None, 0, 0.0):
        base_output.update({
            "edge": 0.0,
            "confidence": 0.0,
            "pick": "PASS",
        })
        return base_output

    odds = features.get("odds", -110)
    try:
        odds = int(odds)
    except (TypeError, ValueError):
        odds = -110
    base_output["odds"] = odds

    if odds_min is not None and odds < odds_min:
        base_output.update({
            "edge": 0.0,
            "confidence": 0.0,
            "pick": "PASS",
            "reason": "ODDS_OUT_OF_RANGE",
        })
        return base_output
    if odds_max is not None and odds > odds_max:
        base_output.update({
            "edge": 0.0,
            "confidence": 0.0,
            "pick": "PASS",
            "reason": "ODDS_OUT_OF_RANGE",
        })
        return base_output

    recent_avg = features.get("recent_avg")
    season_avg = features.get("season_avg")

    if recent_avg in (None, 0, 0.0) and season_avg in (None, 0, 0.0):
        projection = float(line)
    elif season_avg in (None, 0, 0.0):
        projection = float(recent_avg)
    elif recent_avg in (None, 0, 0.0):
        projection = float(season_avg)
    else:
        projection = (float(recent_avg) * 0.6) + (float(season_avg) * 0.4)

    adjustments = {}
    notes = []
    minutes_trend = features.get("minutes_trend")
    if minutes_trend:
        factor = _clamp(float(minutes_trend), 0.85, 1.15)
        projection *= factor
        adjustments["minutes_trend"] = round(factor, 4)
        notes.append("minutes trend")

    usage_avg_5 = features.get("usage_avg_5")
    usage_avg_15 = features.get("usage_avg_15")
    if usage_avg_5 and usage_avg_15:
        factor = _clamp(float(usage_avg_5) / float(usage_avg_15), 0.85, 1.15)
        projection *= factor
        adjustments["usage_trend"] = round(factor, 4)
        notes.append("usage trend")

    expected_usage_factor = features.get("expected_usage_factor")
    expected_usage_notes = features.get("expected_usage_notes")
    if prop_key in _USAGE_SENSITIVE_PROPS and expected_usage_factor not in (None, 0, 0.0):
        try:
            factor = float(expected_usage_factor)
        except (TypeError, ValueError):
            factor = None
        if factor and abs(factor - 1.0) >= 0.01:
            factor = _clamp(factor, 0.85, 1.15)
            projection *= factor
            adjustments["expected_usage"] = round(factor, 4)
            if expected_usage_notes:
                notes.append(f"usage shift ({expected_usage_notes})")
            else:
                notes.append("usage shift")

    team_out_count = features.get("team_out_count") or 0
    team_questionable_count = features.get("team_questionable_count") or 0
    if prop_key in _USAGE_SENSITIVE_PROPS and (team_out_count or team_questionable_count):
        try:
            out_count = int(team_out_count)
        except (TypeError, ValueError):
            out_count = 0
        try:
            q_count = int(team_questionable_count)
        except (TypeError, ValueError):
            q_count = 0
        bump = min(0.08, (0.015 * max(out_count, 0)) + (0.005 * max(q_count, 0)))
        if bump:
            projection *= (1 + bump)
            adjustments["team_usage"] = round(1 + bump, 4)
            notes.append("usage boost (injuries)")

    if injury_status in _RISK_STATUSES:
        projection *= 0.97
        adjustments["injury_risk"] = 0.97
        notes.append("injury risk")
    elif injury_status in _SOFT_STATUSES:
        projection *= 0.99
        adjustments["injury_risk"] = 0.99
        notes.append("injury caution")

    rest_days = features.get("rest_days")
    if rest_days is not None:
        if rest_days <= 1:
            projection *= 0.97
            adjustments["rest"] = 0.97
            notes.append("b2b")
        elif rest_days >= 3:
            projection *= 1.02
            adjustments["rest"] = 1.02
            notes.append("extra rest")

    is_home = features.get("is_home")
    if is_home is True:
        projection *= 1.01
        adjustments["home"] = 1.01
        notes.append("home")

    opp_def = features.get("opp_def_rating")
    league_def = features.get("league_def_avg")
    if opp_def is not None and league_def:
        try:
            opp_def = float(opp_def)
            league_def = float(league_def)
            if opp_def <= league_def - 2:
                projection *= 0.97
                adjustments["opp_def"] = 0.97
                notes.append("tough defense")
            elif opp_def >= league_def + 2:
                projection *= 1.03
                adjustments["opp_def"] = 1.03
                notes.append("soft defense")
        except (TypeError, ValueError):
            pass

    team_pace = features.get("team_pace")
    league_pace = features.get("league_pace_avg")
    if team_pace is not None and league_pace:
        try:
            team_pace = float(team_pace)
            league_pace = float(league_pace)
            if team_pace >= league_pace + 2:
                projection *= 1.02
                adjustments["pace"] = 1.02
                notes.append("fast pace")
            elif team_pace <= league_pace - 2:
                projection *= 0.98
                adjustments["pace"] = 0.98
                notes.append("slow pace")
        except (TypeError, ValueError):
            pass

    # Game Total adjustment (O/U impact on scoring props)
    game_total = features.get("game_total")
    if game_total is not None and prop_key in ("points", "pra", "threes"):
        try:
            total_val = float(game_total)
            league_avg_total = 225.0  # League average O/U
            total_factor = total_val / league_avg_total
            # Apply 30% weight to the deviation
            total_adj = 1 + ((total_factor - 1) * 0.3)
            total_adj = _clamp(total_adj, 0.92, 1.08)
            projection *= total_adj
            adjustments["game_total"] = round(total_adj, 4)
            if total_val >= 235:
                notes.append("high total")
            elif total_val <= 215:
                notes.append("low total")
        except (TypeError, ValueError):
            pass

    # Blowout Risk adjustment (minutes reduction from spread)
    spread = features.get("spread")
    if spread is not None and prop_key in ("points", "rebounds", "assists", "pra", "threes"):
        try:
            spread_val = abs(float(spread))
            if spread_val >= 12:
                projection *= 0.95
                adjustments["blowout_risk"] = 0.95
                notes.append("blowout risk HIGH")
            elif spread_val >= 8:
                projection *= 0.98
                adjustments["blowout_risk"] = 0.98
                notes.append("blowout risk MED")
        except (TypeError, ValueError):
            pass

    # Player vs Team History adjustment
    vs_team_factor = features.get("vs_team_factor")
    if vs_team_factor is not None:
        try:
            factor = float(vs_team_factor)
            factor = _clamp(factor, 0.90, 1.10)
            if abs(factor - 1.0) >= 0.02:
                projection *= factor
                adjustments["vs_team"] = round(factor, 4)
                if factor > 1.0:
                    notes.append("good vs team")
                else:
                    notes.append("poor vs team")
        except (TypeError, ValueError):
            pass

    # True Shooting Efficiency Regression (for points/pra only)
    ts_factor = features.get("ts_factor")
    ts_regression = features.get("ts_regression")
    if ts_factor is not None and prop_key in ("points", "pra"):
        try:
            factor = float(ts_factor)
            factor = _clamp(factor, 0.95, 1.05)
            if abs(factor - 1.0) >= 0.01:
                projection *= factor
                adjustments["ts_regression"] = round(factor, 4)
                if ts_regression == "DOWN":
                    notes.append("TS regress down")
                elif ts_regression == "UP":
                    notes.append("TS regress up")
        except (TypeError, ValueError):
            pass

    recent_std = features.get("recent_std")
    if recent_std:
        try:
            std = max(1.0, float(recent_std))
        except (TypeError, ValueError):
            std = max(1.0, abs(projection) * 0.25)
    else:
        std = max(1.0, abs(projection) * 0.25)
    vol_ratio = _volatility_ratio(recent_std, line)
    if vol_ratio is not None:
        base_output["volatility_ratio"] = round(vol_ratio, 4)

    z = (float(line) - projection) / std
    prob_over = 1 - _normal_cdf(z)

    side = (features.get("side") or "").lower()
    implied = _american_to_implied(odds)
    prob_under = 1 - prob_over

    hit_rate_10 = None
    hit_rate_15 = None
    streak = None
    if side == "over":
        hit_rate_10 = features.get("hit_rate_over_10")
        hit_rate_15 = features.get("hit_rate_over_15")
        streak = features.get("streak_over")
    elif side == "under":
        hit_rate_10 = features.get("hit_rate_under_10")
        hit_rate_15 = features.get("hit_rate_under_15")
        streak = features.get("streak_under")

    if side == "under":
        prob_win = prob_under
        pick = "UNDER"
        edge = prob_win - implied
    elif side == "over":
        prob_win = prob_over
        pick = "OVER"
        edge = prob_win - implied
    else:
        edge_over = prob_over - implied
        edge_under = prob_under - implied
        if edge_over >= edge_under:
            prob_win = prob_over
            pick = "OVER"
            edge = edge_over
        else:
            prob_win = prob_under
            pick = "UNDER"
            edge = edge_under

    raw_model_prob = prob_win
    calibration_meta = None
    model_prob = raw_model_prob
    if calibration:
        model_prob, calibration_meta = apply_calibration(raw_model_prob, prop_key, calibration)

    market_over_odds = features.get("market_over_odds")
    market_under_odds = features.get("market_under_odds")
    over_implied = _american_to_implied(market_over_odds) if market_over_odds is not None else None
    under_implied = _american_to_implied(market_under_odds) if market_under_odds is not None else None
    market_vig = None
    if over_implied is not None and under_implied is not None:
        market_vig = max(0.0, over_implied + under_implied - 1)
        base_output["market_vig"] = round(market_vig, 4)

    market_prob = features.get("market_prob")
    market_prob_val = None
    if market_prob is not None:
        try:
            market_prob_val = float(market_prob)
            prob_win = (model_prob * market_blend) + (market_prob_val * (1 - market_blend))
        except (TypeError, ValueError):
            prob_win = model_prob
    else:
        prob_win = model_prob

    edge = prob_win - implied

    confidence = max(0.2, min(0.95, abs(model_prob - 0.5) * 2))
    n_games = features.get("n_games")
    if n_games is not None:
        try:
            n_games = int(n_games)
            if n_games < 3:
                confidence *= 0.6
            elif n_games < 5:
                confidence *= 0.75
            elif n_games < 8:
                confidence *= 0.85
        except (TypeError, ValueError):
            pass
    vol_penalty = _confidence_volatility_penalty(prop_type, vol_ratio)
    if vol_penalty < 1.0:
        confidence *= vol_penalty
        notes.append("high variance")

    if market_prob_val is None:
        confidence *= 0.92
        notes.append("market missing")
    if market_vig is not None and market_vig > 0.08:
        confidence *= 0.9
        notes.append("high vig")
    if market_prob_val is not None:
        if abs(model_prob - market_prob_val) >= 0.2:
            confidence *= 0.9
            notes.append("market divergence")

    if injury_status in _RISK_STATUSES:
        confidence = max(0.05, confidence - 0.1)
    elif injury_status in _SOFT_STATUSES:
        confidence = max(0.05, confidence - 0.05)

    if injury_status in _RISK_STATUSES:
        edge *= max(0.0, float(injury_risk_edge_multiplier))
        confidence *= max(0.0, float(injury_risk_confidence_multiplier))
        notes.append("injury risk penalty")

    if (
        odds_confidence_min is not None
        and odds_confidence_max is not None
        and confidence_cap_outside_band is not None
    ):
        if odds < odds_confidence_min or odds > odds_confidence_max:
            cap = max(0.05, min(0.95, float(confidence_cap_outside_band)))
            if confidence > cap:
                confidence = cap
                notes.append("confidence cap (odds band)")

    thresholds = _adjust_thresholds(min_edge, min_confidence, prop_type, recent_std, line)
    edge_threshold = thresholds["edge"]
    confidence_threshold = thresholds["confidence"]
    base_output["edge_threshold"] = round(edge_threshold, 4)
    base_output["confidence_threshold"] = round(confidence_threshold, 4)
    base_output["risk_factor"] = round(_risk_factor(prop_type), 3)
    base_output["volatility_factor"] = round(_volatility_factor(recent_std, line), 3)

    if edge < edge_threshold or confidence < confidence_threshold:
        pick = "PASS"

    usage_note_parts = []
    usage_avg_5 = features.get("usage_avg_5")
    usage_avg_15 = features.get("usage_avg_15")
    if usage_avg_5 and usage_avg_15:
        try:
            usage_ratio = float(usage_avg_5) / float(usage_avg_15)
            if usage_ratio >= 1.05:
                usage_note_parts.append("Usage up")
            elif usage_ratio <= 0.95:
                usage_note_parts.append("Usage down")
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    if expected_usage_notes:
        usage_note_parts.append(f"Expected usage: {expected_usage_notes}")

    team_out = features.get("team_out_count") or 0
    team_questionable = features.get("team_questionable_count") or 0
    if team_out:
        usage_note_parts.append(f"Teammates OUT: {team_out}")
    if team_questionable:
        usage_note_parts.append(f"Teammates Q: {team_questionable}")
    if injury_status:
        usage_note_parts.append(f"Player status: {injury_status}")

    usage_expectation = "; ".join(usage_note_parts) if usage_note_parts else None

    return {
        "prop_id": row.get("prop_id"),
        "player_id": features.get("player_id"),
        "player_name": features.get("player_name"),
        "prop_type": prop_type,
        "side": features.get("side"),
        "line": line,
        "event_id": features.get("event_id"),
        "home_team": features.get("home_team"),
        "away_team": features.get("away_team"),
        "game_time": features.get("game_time"),
        "player_team": features.get("player_team"),
        "opponent_team": features.get("opponent_team"),
        "is_home": features.get("is_home"),
        "rest_days": features.get("rest_days"),
        "b2b": features.get("b2b"),
        "projection": round(projection, 2),
        "probability": round(prob_win, 4),
        "raw_model_probability": round(raw_model_prob, 4),
        "model_probability": round(model_prob, 4),
        "market_probability": round(float(market_prob), 4) if market_prob is not None else None,
        "market_prob": round(float(market_prob), 4) if market_prob is not None else None,
        "market_over_odds": features.get("market_over_odds"),
        "market_under_odds": features.get("market_under_odds"),
        "edge": round(edge, 4),
        "confidence": round(confidence, 4),
        "confidence_tier": _confidence_tier(confidence),
        "adjustments": adjustments or None,
        "adjustment_notes": ", ".join(notes) if notes else None,
        "avg_5": features.get("recent_avg"),
        "avg_10": features.get("avg_10"),
        "avg_15": features.get("avg_15"),
        "trend": features.get("trend"),
        "recent_std": features.get("recent_std"),
        "recent_hit_rate_10": hit_rate_10,
        "recent_hit_rate_15": hit_rate_15,
        "recent_streak": streak,
        "team_out_count": features.get("team_out_count"),
        "team_questionable_count": features.get("team_questionable_count"),
        "team_probable_count": features.get("team_probable_count"),
        "team_out_detail": features.get("team_out_detail"),
        "team_questionable_detail": features.get("team_questionable_detail"),
        "team_probable_detail": features.get("team_probable_detail"),
        "usage_expectation": usage_expectation,
        "expected_usage_factor": features.get("expected_usage_factor"),
        "expected_usage_notes": features.get("expected_usage_notes"),
        "pick": pick,
        "odds": odds,
        "market_vig": base_output.get("market_vig"),
        "edge_threshold": base_output.get("edge_threshold"),
        "confidence_threshold": base_output.get("confidence_threshold"),
        "risk_factor": base_output.get("risk_factor"),
        "volatility_factor": base_output.get("volatility_factor"),
        "volatility_ratio": base_output.get("volatility_ratio"),
        "odds_min": odds_min,
        "odds_max": odds_max,
        "odds_confidence_min": odds_confidence_min,
        "odds_confidence_max": odds_confidence_max,
        "confidence_cap_outside_band": confidence_cap_outside_band,
        "reason": base_output.get("reason"),
        "calibration_source": calibration_meta.get("source") if calibration_meta else None,
        "calibration_slope": calibration_meta.get("slope") if calibration_meta else None,
        "calibration_intercept": calibration_meta.get("intercept") if calibration_meta else None,
        "injury_status": injury_status or None,
        "injury_source": injury_source,
        "injury_detail": injury_detail,
    }
