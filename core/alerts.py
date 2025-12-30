"""Alert system for real-time notifications.

This module provides multiple alert channels for notifying users about
significant events like line movements, high-edge opportunities, and
injury updates.

Usage:
    from core.alerts import AlertManager, ConsoleAlert, WebhookAlert

    manager = AlertManager()
    manager.add_handler(ConsoleAlert())
    manager.add_handler(WebhookAlert("https://hooks.slack.com/..."))

    manager.send_line_movement(movement)
    manager.send_high_edge_alert(analysis)
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Base alert structure."""
    alert_type: str  # 'line_movement', 'high_edge', 'injury', 'steam_move'
    title: str
    message: str
    priority: str = "normal"  # 'low', 'normal', 'high', 'urgent'
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type,
            'title': self.title,
            'message': self.message,
            'priority': self.priority,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
        }


class AlertHandler(ABC):
    """Abstract base class for alert handlers."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """
        Send an alert through this handler.

        Returns:
            True if alert was sent successfully
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name for logging."""
        pass


class ConsoleAlert(AlertHandler):
    """Print alerts to console with formatting."""

    # Priority indicators
    PRIORITY_ICONS = {
        'low': '○',
        'normal': '●',
        'high': '◉',
        'urgent': '🔴',
    }

    # Alert type icons
    TYPE_ICONS = {
        'line_movement': '📊',
        'high_edge': '💰',
        'injury': '🏥',
        'steam_move': '🔥',
        'system': '⚙️',
    }

    def __init__(self, show_data: bool = False):
        """
        Initialize console alert handler.

        Args:
            show_data: Whether to print full alert data
        """
        self.show_data = show_data

    @property
    def name(self) -> str:
        return "console"

    def send(self, alert: Alert) -> bool:
        try:
            icon = self.TYPE_ICONS.get(alert.alert_type, '📌')
            priority_icon = self.PRIORITY_ICONS.get(alert.priority, '●')
            timestamp = alert.timestamp.strftime("%H:%M:%S")

            print(f"\n{priority_icon} [{timestamp}] {icon} {alert.title}")
            print(f"   {alert.message}")

            if self.show_data and alert.data:
                print(f"   Data: {json.dumps(alert.data, indent=2)}")

            return True
        except (IOError, OSError) as e:
            logger.error(f"Console alert I/O error: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Console alert formatting error: {e}")
            return False
        except Exception as e:
            logger.error(f"Console alert failed: {e}")
            return False


class LoggingAlert(AlertHandler):
    """Send alerts to Python logging system."""

    PRIORITY_LEVELS = {
        'low': logging.DEBUG,
        'normal': logging.INFO,
        'high': logging.WARNING,
        'urgent': logging.ERROR,
    }

    def __init__(self, logger_name: str = "alerts"):
        self._logger = logging.getLogger(logger_name)

    @property
    def name(self) -> str:
        return "logging"

    def send(self, alert: Alert) -> bool:
        try:
            level = self.PRIORITY_LEVELS.get(alert.priority, logging.INFO)
            self._logger.log(
                level,
                f"[{alert.alert_type}] {alert.title}: {alert.message}",
                extra={'alert_data': alert.data}
            )
            return True
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Logging alert formatting error: {e}")
            return False
        except Exception as e:
            logger.error(f"Logging alert failed: {e}")
            return False


class WebhookAlert(AlertHandler):
    """Send alerts to a webhook URL (Slack, Discord, etc.)."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10
    ):
        """
        Initialize webhook handler.

        Args:
            url: Webhook URL to POST to
            headers: Optional headers (auth, content-type)
            timeout: Request timeout in seconds
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"webhook:{self.url[:30]}..."

    def send(self, alert: Alert) -> bool:
        try:
            import requests

            payload = self._format_payload(alert)
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except ImportError:
            logger.error("requests library required for webhook alerts")
            return False
        except requests.RequestException as e:
            logger.error(f"Webhook alert network error: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Webhook alert payload formatting error: {e}")
            return False
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            return False

    def _format_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format alert for generic webhook. Override for specific platforms."""
        return {
            'text': f"*{alert.title}*\n{alert.message}",
            'alert': alert.to_dict()
        }


class SlackWebhookAlert(WebhookAlert):
    """Slack-formatted webhook alerts."""

    PRIORITY_COLORS = {
        'low': '#808080',      # gray
        'normal': '#36a64f',   # green
        'high': '#ff9800',     # orange
        'urgent': '#ff0000',   # red
    }

    def _format_payload(self, alert: Alert) -> Dict[str, Any]:
        color = self.PRIORITY_COLORS.get(alert.priority, '#36a64f')

        return {
            'attachments': [{
                'color': color,
                'title': alert.title,
                'text': alert.message,
                'footer': f"NBA Props Alert | {alert.alert_type}",
                'ts': int(alert.timestamp.timestamp()),
                'fields': self._format_fields(alert.data) if alert.data else [],
            }]
        }

    def _format_fields(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert data dict to Slack fields."""
        fields = []
        for key, value in data.items():
            if key in ('player', 'prop_type', 'line', 'edge', 'direction'):
                fields.append({
                    'title': key.replace('_', ' ').title(),
                    'value': str(value),
                    'short': True
                })
        return fields


class FileAlert(AlertHandler):
    """Append alerts to a JSON lines file."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    @property
    def name(self) -> str:
        return f"file:{self.filepath}"

    def send(self, alert: Alert) -> bool:
        try:
            with open(self.filepath, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
            return True
        except (IOError, OSError, PermissionError) as e:
            logger.error(f"File alert I/O error: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"File alert serialization error: {e}")
            return False
        except Exception as e:
            logger.error(f"File alert failed: {e}")
            return False


class CallbackAlert(AlertHandler):
    """Custom callback function for alerts."""

    def __init__(self, callback, handler_name: str = "callback"):
        """
        Initialize callback handler.

        Args:
            callback: Function that takes an Alert and returns bool
            handler_name: Name for this handler
        """
        self.callback = callback
        self._name = handler_name

    @property
    def name(self) -> str:
        return self._name

    def send(self, alert: Alert) -> bool:
        try:
            return bool(self.callback(alert))
        except (TypeError, AttributeError) as e:
            logger.error(f"Callback alert invocation error: {e}")
            return False
        except Exception as e:
            logger.error(f"Callback alert failed: {e}")
            return False


class AlertManager:
    """
    Manages multiple alert handlers and provides convenience methods.

    Usage:
        manager = AlertManager()
        manager.add_handler(ConsoleAlert())
        manager.add_handler(SlackWebhookAlert("https://hooks.slack.com/..."))

        # Send alerts
        manager.send_line_movement(movement)
        manager.send_high_edge(analysis)
    """

    def __init__(self):
        self._handlers: List[AlertHandler] = []
        self._alert_history: List[Alert] = []
        self._max_history = 500

    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)
        logger.info(f"Added alert handler: {handler.name}")

    def remove_handler(self, handler_name: str) -> bool:
        """Remove a handler by name."""
        for i, h in enumerate(self._handlers):
            if h.name == handler_name:
                self._handlers.pop(i)
                logger.info(f"Removed alert handler: {handler_name}")
                return True
        return False

    def send(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert to all handlers.

        Returns:
            Dict mapping handler name to success status
        """
        results = {}

        for handler in self._handlers:
            try:
                results[handler.name] = handler.send(alert)
            except Exception as e:
                logger.error(f"Handler {handler.name} failed: {e}")
                results[handler.name] = False

        # Track history
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

        return results

    def send_line_movement(
        self,
        movement,  # LineMovement object
        priority: str = "normal"
    ) -> Dict[str, bool]:
        """Send a line movement alert."""
        direction = "⬆️" if movement.direction == "UP" else "⬇️"

        alert = Alert(
            alert_type='line_movement',
            title=f"Line Movement: {movement.player}",
            message=(
                f"{movement.prop_type}: {movement.old_line} → {movement.new_line} "
                f"{direction} ({movement.movement_pct:+.1%})"
            ),
            priority=priority,
            data=movement.to_dict() if hasattr(movement, 'to_dict') else None
        )
        return self.send(alert)

    def send_steam_move(
        self,
        steam_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Send a steam move alert (multi-book coordinated movement)."""
        direction = "⬆️" if steam_data['direction'] == "UP" else "⬇️"
        books = ", ".join(steam_data['books'][:3])
        if len(steam_data['books']) > 3:
            books += f" +{len(steam_data['books']) - 3} more"

        alert = Alert(
            alert_type='steam_move',
            title=f"🔥 Steam Move: {steam_data['player']}",
            message=(
                f"{steam_data['prop_type']} moving {direction} across {len(steam_data['books'])} books: {books}\n"
                f"Avg movement: {steam_data['avg_movement']:+.1%}"
            ),
            priority='high',
            data=steam_data
        )
        return self.send(alert)

    def send_high_edge(
        self,
        analysis,  # PropAnalysis object
        min_edge: float = 0.10
    ) -> Dict[str, bool]:
        """Send alert for high-edge opportunity."""
        edge = getattr(analysis, 'edge', 0)
        if edge < min_edge:
            return {}

        priority = 'urgent' if edge >= 0.15 else 'high'

        alert = Alert(
            alert_type='high_edge',
            title=f"💰 High Edge: {analysis.player_name}",
            message=(
                f"{analysis.prop_type} {analysis.pick} {analysis.line}\n"
                f"Projection: {analysis.projection:.1f} | Edge: {edge:.1%} | "
                f"Confidence: {analysis.confidence:.0%}"
            ),
            priority=priority,
            data=analysis.to_dict() if hasattr(analysis, 'to_dict') else None
        )
        return self.send(alert)

    def send_injury_update(
        self,
        player: str,
        status: str,
        impact: Optional[str] = None
    ) -> Dict[str, bool]:
        """Send injury status update alert."""
        alert = Alert(
            alert_type='injury',
            title=f"🏥 Injury Update: {player}",
            message=f"Status: {status}" + (f"\nImpact: {impact}" if impact else ""),
            priority='high' if status in ('OUT', 'DOUBTFUL') else 'normal',
            data={'player': player, 'status': status, 'impact': impact}
        )
        return self.send(alert)

    def send_system(
        self,
        title: str,
        message: str,
        priority: str = "low"
    ) -> Dict[str, bool]:
        """Send system notification."""
        alert = Alert(
            alert_type='system',
            title=title,
            message=message,
            priority=priority
        )
        return self.send(alert)

    def get_history(
        self,
        alert_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Alert]:
        """Get recent alert history."""
        history = self._alert_history

        if alert_type:
            history = [a for a in history if a.alert_type == alert_type]

        return history[-limit:]

    @property
    def handler_count(self) -> int:
        """Number of active handlers."""
        return len(self._handlers)

    @property
    def handler_names(self) -> List[str]:
        """List of active handler names."""
        return [h.name for h in self._handlers]


def create_default_manager(
    console: bool = True,
    log_file: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> AlertManager:
    """
    Create an AlertManager with common handlers.

    Args:
        console: Enable console output
        log_file: Path to JSON log file (optional)
        webhook_url: Slack/Discord webhook URL (optional)

    Returns:
        Configured AlertManager
    """
    manager = AlertManager()

    if console:
        manager.add_handler(ConsoleAlert())

    if log_file:
        manager.add_handler(FileAlert(log_file))

    if webhook_url:
        if 'slack' in webhook_url.lower():
            manager.add_handler(SlackWebhookAlert(webhook_url))
        else:
            manager.add_handler(WebhookAlert(webhook_url))

    return manager
