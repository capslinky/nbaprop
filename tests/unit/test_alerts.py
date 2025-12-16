"""Unit tests for alert system functionality."""

import pytest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from core.alerts import (
    Alert,
    AlertManager,
    ConsoleAlert,
    LoggingAlert,
    FileAlert,
    CallbackAlert,
    WebhookAlert,
    SlackWebhookAlert,
    create_default_manager,
)
from core.line_monitor import LineMovement


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test basic Alert creation."""
        alert = Alert(
            alert_type='test',
            title='Test Alert',
            message='This is a test message'
        )

        assert alert.alert_type == 'test'
        assert alert.title == 'Test Alert'
        assert alert.message == 'This is a test message'
        assert alert.priority == 'normal'
        assert alert.timestamp is not None

    def test_alert_with_data(self):
        """Test Alert with data payload."""
        alert = Alert(
            alert_type='high_edge',
            title='High Edge',
            message='Found value',
            priority='high',
            data={'player': 'Test', 'edge': 0.12}
        )

        assert alert.data['player'] == 'Test'
        assert alert.data['edge'] == 0.12

    def test_to_dict(self):
        """Test Alert serialization."""
        alert = Alert(
            alert_type='line_movement',
            title='Line Moved',
            message='Big move',
            priority='urgent'
        )

        data = alert.to_dict()
        assert data['alert_type'] == 'line_movement'
        assert data['priority'] == 'urgent'
        assert 'timestamp' in data


class TestConsoleAlert:
    """Tests for ConsoleAlert handler."""

    def test_console_alert_creation(self):
        """Test ConsoleAlert initialization."""
        handler = ConsoleAlert()
        assert handler.name == "console"

    def test_console_alert_send(self, capsys):
        """Test sending alert to console."""
        handler = ConsoleAlert()

        alert = Alert(
            alert_type='line_movement',
            title='Test Movement',
            message='Player A points moved'
        )

        result = handler.send(alert)
        assert result is True

        captured = capsys.readouterr()
        assert 'Test Movement' in captured.out
        assert 'Player A points moved' in captured.out

    def test_console_alert_with_data(self, capsys):
        """Test console alert showing data."""
        handler = ConsoleAlert(show_data=True)

        alert = Alert(
            alert_type='test',
            title='Test',
            message='Message',
            data={'key': 'value'}
        )

        handler.send(alert)
        captured = capsys.readouterr()
        assert 'Data:' in captured.out


class TestLoggingAlert:
    """Tests for LoggingAlert handler."""

    def test_logging_alert_creation(self):
        """Test LoggingAlert initialization."""
        handler = LoggingAlert()
        assert handler.name == "logging"

    def test_logging_alert_send(self, caplog):
        """Test sending alert to logger."""
        handler = LoggingAlert(logger_name="test_alerts")

        alert = Alert(
            alert_type='test',
            title='Test Title',
            message='Test Message'
        )

        with caplog.at_level('INFO'):
            result = handler.send(alert)

        assert result is True


class TestFileAlert:
    """Tests for FileAlert handler."""

    def test_file_alert_creation(self):
        """Test FileAlert initialization."""
        handler = FileAlert("/tmp/test.json")
        assert "file:" in handler.name

    def test_file_alert_send(self):
        """Test appending alert to file."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            filepath = f.name

        try:
            handler = FileAlert(filepath)

            alert = Alert(
                alert_type='test',
                title='Test',
                message='Test Message'
            )

            result = handler.send(alert)
            assert result is True

            # Verify file content
            with open(filepath) as f:
                line = f.readline()
                data = json.loads(line)
                assert data['alert_type'] == 'test'
                assert data['title'] == 'Test'
        finally:
            os.unlink(filepath)

    def test_file_alert_appends(self):
        """Test that multiple alerts are appended."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            filepath = f.name

        try:
            handler = FileAlert(filepath)

            for i in range(3):
                alert = Alert(
                    alert_type='test',
                    title=f'Alert {i}',
                    message='Message'
                )
                handler.send(alert)

            # Verify 3 lines
            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) == 3
        finally:
            os.unlink(filepath)


class TestCallbackAlert:
    """Tests for CallbackAlert handler."""

    def test_callback_alert_creation(self):
        """Test CallbackAlert initialization."""
        callback = MagicMock(return_value=True)
        handler = CallbackAlert(callback, handler_name="test_callback")
        assert handler.name == "test_callback"

    def test_callback_alert_send(self):
        """Test callback is invoked with alert."""
        callback = MagicMock(return_value=True)
        handler = CallbackAlert(callback)

        alert = Alert(
            alert_type='test',
            title='Test',
            message='Message'
        )

        result = handler.send(alert)
        assert result is True
        callback.assert_called_once_with(alert)

    def test_callback_alert_returns_callback_result(self):
        """Test callback return value is used."""
        callback = MagicMock(return_value=False)
        handler = CallbackAlert(callback)

        alert = Alert(alert_type='test', title='Test', message='Message')
        result = handler.send(alert)
        assert result is False


class TestWebhookAlert:
    """Tests for WebhookAlert handler."""

    def test_webhook_alert_creation(self):
        """Test WebhookAlert initialization."""
        handler = WebhookAlert("https://example.com/webhook")
        assert "webhook:" in handler.name

    @patch('requests.post')
    def test_webhook_alert_send(self, mock_post):
        """Test sending alert to webhook."""
        mock_post.return_value.raise_for_status = MagicMock()

        handler = WebhookAlert("https://example.com/webhook")
        alert = Alert(alert_type='test', title='Test', message='Message')

        result = handler.send(alert)
        assert result is True
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_webhook_alert_failure(self, mock_post):
        """Test webhook failure handling."""
        mock_post.side_effect = Exception("Connection failed")

        handler = WebhookAlert("https://example.com/webhook")
        alert = Alert(alert_type='test', title='Test', message='Message')

        result = handler.send(alert)
        assert result is False


class TestSlackWebhookAlert:
    """Tests for SlackWebhookAlert handler."""

    def test_slack_payload_format(self):
        """Test Slack-specific payload formatting."""
        handler = SlackWebhookAlert("https://hooks.slack.com/test")

        alert = Alert(
            alert_type='line_movement',
            title='Line Moved',
            message='Test message',
            priority='high',
            data={'player': 'Test', 'edge': 0.12}
        )

        payload = handler._format_payload(alert)

        assert 'attachments' in payload
        assert len(payload['attachments']) == 1
        assert payload['attachments'][0]['title'] == 'Line Moved'
        assert 'fields' in payload['attachments'][0]


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_manager_creation(self):
        """Test AlertManager initialization."""
        manager = AlertManager()
        assert manager.handler_count == 0

    def test_add_handler(self):
        """Test adding handlers."""
        manager = AlertManager()
        manager.add_handler(ConsoleAlert())

        assert manager.handler_count == 1
        assert 'console' in manager.handler_names

    def test_remove_handler(self):
        """Test removing handlers."""
        manager = AlertManager()
        manager.add_handler(ConsoleAlert())

        removed = manager.remove_handler('console')
        assert removed is True
        assert manager.handler_count == 0

    def test_remove_nonexistent_handler(self):
        """Test removing handler that doesn't exist."""
        manager = AlertManager()
        removed = manager.remove_handler('nonexistent')
        assert removed is False

    def test_send_to_all_handlers(self):
        """Test sending alert to all handlers."""
        manager = AlertManager()

        callback1 = MagicMock(return_value=True)
        callback2 = MagicMock(return_value=True)

        manager.add_handler(CallbackAlert(callback1, "cb1"))
        manager.add_handler(CallbackAlert(callback2, "cb2"))

        alert = Alert(alert_type='test', title='Test', message='Message')
        results = manager.send(alert)

        assert results['cb1'] is True
        assert results['cb2'] is True
        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_send_line_movement(self):
        """Test convenience method for line movements."""
        manager = AlertManager()
        callback = MagicMock(return_value=True)
        manager.add_handler(CallbackAlert(callback, "test"))

        movement = LineMovement(
            player="Test Player",
            prop_type="points",
            old_line=25.0,
            new_line=27.0,
            movement_pct=0.08
        )

        results = manager.send_line_movement(movement)
        assert results['test'] is True

        # Check the alert was formatted correctly
        call_args = callback.call_args[0][0]
        assert call_args.alert_type == 'line_movement'
        assert 'Test Player' in call_args.title

    def test_send_steam_move(self):
        """Test convenience method for steam moves."""
        manager = AlertManager()
        callback = MagicMock(return_value=True)
        manager.add_handler(CallbackAlert(callback, "test"))

        steam_data = {
            'player': 'Test Player',
            'prop_type': 'points',
            'direction': 'UP',
            'books': ['fanduel', 'draftkings', 'betmgm'],
            'avg_movement': 0.08
        }

        results = manager.send_steam_move(steam_data)
        assert results['test'] is True

        call_args = callback.call_args[0][0]
        assert call_args.alert_type == 'steam_move'
        assert call_args.priority == 'high'

    def test_send_injury_update(self):
        """Test injury update alert."""
        manager = AlertManager()
        callback = MagicMock(return_value=True)
        manager.add_handler(CallbackAlert(callback, "test"))

        results = manager.send_injury_update(
            player="LeBron James",
            status="OUT",
            impact="Season-ending injury"
        )

        assert results['test'] is True
        call_args = callback.call_args[0][0]
        assert call_args.alert_type == 'injury'
        assert call_args.priority == 'high'

    def test_get_history(self):
        """Test alert history retrieval."""
        manager = AlertManager()
        callback = MagicMock(return_value=True)
        manager.add_handler(CallbackAlert(callback, "test"))

        # Send a few alerts
        manager.send_system("System 1", "Message 1")
        manager.send_system("System 2", "Message 2")

        history = manager.get_history()
        assert len(history) == 2

        # Filter by type
        system_history = manager.get_history(alert_type='system')
        assert len(system_history) == 2

    def test_handler_failure_isolated(self):
        """Test that one handler failure doesn't affect others."""
        manager = AlertManager()

        failing = MagicMock(side_effect=Exception("Failed"))
        success = MagicMock(return_value=True)

        manager.add_handler(CallbackAlert(failing, "failing"))
        manager.add_handler(CallbackAlert(success, "success"))

        alert = Alert(alert_type='test', title='Test', message='Message')
        results = manager.send(alert)

        assert results['failing'] is False
        assert results['success'] is True


class TestCreateDefaultManager:
    """Tests for create_default_manager factory."""

    def test_create_with_console_only(self):
        """Test creating manager with console output."""
        manager = create_default_manager(console=True)
        assert manager.handler_count == 1
        assert 'console' in manager.handler_names

    def test_create_with_file(self):
        """Test creating manager with file logging."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            filepath = f.name

        try:
            manager = create_default_manager(
                console=False,
                log_file=filepath
            )
            assert manager.handler_count == 1
            assert any('file:' in name for name in manager.handler_names)
        finally:
            os.unlink(filepath)

    def test_create_with_all_options(self):
        """Test creating manager with all options."""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            filepath = f.name

        try:
            manager = create_default_manager(
                console=True,
                log_file=filepath,
                webhook_url="https://example.com/webhook"
            )
            assert manager.handler_count == 3
        finally:
            os.unlink(filepath)

    def test_slack_webhook_detection(self):
        """Test that Slack webhooks get SlackWebhookAlert."""
        manager = create_default_manager(
            console=False,
            webhook_url="https://hooks.slack.com/services/xxx"
        )
        assert manager.handler_count == 1
        # Handler should be SlackWebhookAlert type
