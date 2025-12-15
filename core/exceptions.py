"""
Custom exceptions for the NBA prop analysis system.

Provides a hierarchy of exceptions for better error handling
and debugging compared to silent failures.

Usage:
    from core.exceptions import PlayerNotFoundError, DataFetchError

    try:
        logs = fetcher.get_player_game_logs("Invalid Player")
    except PlayerNotFoundError as e:
        print(f"Player not found: {e}")
    except DataFetchError as e:
        print(f"Data fetch failed: {e}")
"""


class NBAPropError(Exception):
    """
    Base exception for all NBA prop analysis errors.

    All custom exceptions inherit from this, allowing:
        except NBAPropError:
            # Catch any system error
    """
    pass


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataFetchError(NBAPropError):
    """
    Error fetching data from an API or data source.

    Raised when:
    - API request fails (network error, timeout)
    - API returns error status code
    - Data source is unavailable
    """

    def __init__(self, source: str, message: str = None, original_error: Exception = None):
        self.source = source
        self.original_error = original_error
        msg = f"Error fetching from {source}"
        if message:
            msg += f": {message}"
        if original_error:
            msg += f" (caused by: {type(original_error).__name__}: {original_error})"
        super().__init__(msg)


class PlayerNotFoundError(NBAPropError):
    """
    Player not found in the database or API.

    Raised when:
    - Player name doesn't match any known player
    - Player ID lookup fails
    - Player has no data for requested season
    """

    def __init__(self, player_name: str, season: str = None):
        self.player_name = player_name
        self.season = season
        msg = f"Player not found: {player_name}"
        if season:
            msg += f" (season: {season})"
        super().__init__(msg)


class InsufficientDataError(NBAPropError):
    """
    Not enough historical data for reliable analysis.

    Raised when:
    - Player has fewer games than MIN_SAMPLE_SIZE
    - No recent games in lookback period
    - Missing required data fields
    """

    def __init__(self, player_name: str, games_found: int, games_required: int):
        self.player_name = player_name
        self.games_found = games_found
        self.games_required = games_required
        super().__init__(
            f"Insufficient data for {player_name}: "
            f"found {games_found} games, need {games_required}"
        )


# =============================================================================
# API ERRORS
# =============================================================================

class OddsAPIError(NBAPropError):
    """
    Error from The Odds API.

    Raised when:
    - API key is invalid or missing
    - Request quota exceeded
    - API returns error response
    """

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        self.status_code = status_code
        self.response_body = response_body
        msg = f"Odds API error: {message}"
        if status_code:
            msg += f" (status: {status_code})"
        super().__init__(msg)


class RateLimitError(OddsAPIError):
    """
    Rate limit exceeded on an API.

    Raised when:
    - Too many requests in time window
    - Monthly quota exhausted
    """

    def __init__(self, api_name: str, retry_after: int = None):
        self.api_name = api_name
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for {api_name}"
        if retry_after:
            msg += f" (retry after {retry_after} seconds)"
        super().__init__(msg)


# =============================================================================
# ANALYSIS ERRORS
# =============================================================================

class AnalysisError(NBAPropError):
    """
    Error during prop analysis.

    Raised when:
    - Invalid prop type specified
    - Analysis calculation fails
    - Required context data missing
    """

    def __init__(self, player_name: str, prop_type: str, reason: str):
        self.player_name = player_name
        self.prop_type = prop_type
        self.reason = reason
        super().__init__(f"Analysis failed for {player_name} {prop_type}: {reason}")


class InvalidPropTypeError(AnalysisError):
    """
    Invalid or unsupported prop type.

    Valid prop types: points, rebounds, assists, pra, threes, steals, blocks, turnovers
    """

    VALID_PROP_TYPES = ['points', 'rebounds', 'assists', 'pra', 'threes', 'steals', 'blocks', 'turnovers']

    def __init__(self, prop_type: str):
        self.prop_type = prop_type
        super().__init__(
            player_name="",
            prop_type=prop_type,
            reason=f"Invalid prop type '{prop_type}'. Valid types: {', '.join(self.VALID_PROP_TYPES)}"
        )


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(NBAPropError):
    """
    Configuration or setup error.

    Raised when:
    - Required API key missing
    - Invalid configuration value
    - Required dependency not installed
    """

    def __init__(self, setting: str, message: str):
        self.setting = setting
        super().__init__(f"Configuration error ({setting}): {message}")
