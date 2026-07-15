"""Tests for bedrock.profile config.yaml support.

Verifies fix for #43143: bedrock.profile in config.yaml was ignored;
the adapters used boto3.client() directly instead of
boto3.Session(profile_name=...).client().

Tests cover:
  - bedrock_adapter: _get_bedrock_runtime_client uses configured profile
  - bedrock_adapter: _get_bedrock_control_client uses configured profile
  - anthropic_adapter: build_anthropic_bedrock_client passes aws_profile
  - No profile configured: falls back to default credential chain
  - Profile cache key includes profile to avoid stale credentials
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _mock_boto3_session():
    """Patch boto3 and capture Session creation."""
    mock_boto3 = MagicMock()
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_session.client.return_value = mock_client
    mock_boto3.Session.return_value = mock_session
    mock_boto3.client.return_value = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        yield mock_boto3, mock_session, mock_client


# ---------------------------------------------------------------------------
# bedrock_adapter: _get_bedrock_runtime_client
# ---------------------------------------------------------------------------

class TestGetBedrockRuntimeClientProfile:
    """_get_bedrock_runtime_client respects bedrock.profile from config."""

    def test_uses_session_with_profile_when_configured(self):
        """When bedrock.profile is set, creates a boto3.Session with that profile."""
        from agent.bedrock_adapter import _get_bedrock_runtime_client, reset_client_cache
        reset_client_cache()
        try:
            with _mock_boto3_session() as (mock_boto3, mock_session, mock_client):
                with patch("agent.bedrock_adapter._load_bedrock_config",
                           return_value={"profile": "my-aws-profile"}):
                    result = _get_bedrock_runtime_client("us-east-1")

                mock_boto3.Session.assert_called_once_with(profile_name="my-aws-profile")
                mock_session.client.assert_called_once_with(
                    "bedrock-runtime", region_name="us-east-1"
                )
                assert result is mock_client
        finally:
            reset_client_cache()

    def test_uses_default_chain_when_no_profile(self):
        """When no bedrock.profile, uses boto3.client() directly (default chain)."""
        from agent.bedrock_adapter import _get_bedrock_runtime_client, reset_client_cache
        reset_client_cache()
        try:
            with _mock_boto3_session() as (mock_boto3, mock_session, mock_client):
                with patch("agent.bedrock_adapter._load_bedrock_config",
                           return_value={"region": "us-east-1"}):
                    result = _get_bedrock_runtime_client("us-east-1")

                mock_boto3.Session.assert_not_called()
                mock_boto3.client.assert_called_once_with(
                    "bedrock-runtime", region_name="us-east-1"
                )
                assert result is mock_client
        finally:
            reset_client_cache()

    def test_empty_profile_uses_default_chain(self):
        """Empty bedrock.profile string falls back to default chain."""
        from agent.bedrock_adapter import _get_bedrock_runtime_client, reset_client_cache
        reset_client_cache()
        try:
            with _mock_boto3_session() as (mock_boto3, mock_session, mock_client):
                with patch("agent.bedrock_adapter._load_bedrock_config",
                           return_value={"profile": "  "}):
                    result = _get_bedrock_runtime_client("us-east-1")

                mock_boto3.Session.assert_not_called()
                mock_boto3.client.assert_called_once_with(
                    "bedrock-runtime", region_name="us-east-1"
                )
                assert result is mock_client
        finally:
            reset_client_cache()

    def test_profile_is_part_of_cache_key(self):
        """A profile change for the same region creates a fresh client."""
        from agent.bedrock_adapter import _get_bedrock_runtime_client, reset_client_cache
        reset_client_cache()
        try:
            with _mock_boto3_session() as (mock_boto3, mock_session, _mock_client):
                first_client = MagicMock(name="first-client")
                second_client = MagicMock(name="second-client")
                mock_session.client.side_effect = [first_client, second_client]
                with patch("agent.bedrock_adapter._load_bedrock_config",
                           side_effect=[{"profile": "dev"}, {"profile": "prod"}]):
                    first = _get_bedrock_runtime_client("us-east-1")
                    second = _get_bedrock_runtime_client("us-east-1")

                assert first is first_client
                assert second is second_client
                assert mock_boto3.Session.call_args_list[0].kwargs == {"profile_name": "dev"}
                assert mock_boto3.Session.call_args_list[1].kwargs == {"profile_name": "prod"}
        finally:
            reset_client_cache()


# ---------------------------------------------------------------------------
# bedrock_adapter: _get_bedrock_control_client
# ---------------------------------------------------------------------------

class TestGetBedrockControlClientProfile:
    """_get_bedrock_control_client respects bedrock.profile from config."""

    def test_uses_session_with_profile_when_configured(self):
        """When bedrock.profile is set, creates a boto3.Session with that profile."""
        from agent.bedrock_adapter import _get_bedrock_control_client, reset_client_cache
        reset_client_cache()
        try:
            with _mock_boto3_session() as (mock_boto3, mock_session, mock_client):
                with patch("agent.bedrock_adapter._load_bedrock_config",
                           return_value={"profile": "prod-profile"}):
                    result = _get_bedrock_control_client("eu-west-1")

                mock_boto3.Session.assert_called_once_with(profile_name="prod-profile")
                mock_session.client.assert_called_once_with(
                    "bedrock", region_name="eu-west-1"
                )
                assert result is mock_client
        finally:
            reset_client_cache()


# ---------------------------------------------------------------------------
# anthropic_adapter: build_anthropic_bedrock_client
# ---------------------------------------------------------------------------

class TestBuildAnthropicBedrockClientProfile:
    """build_anthropic_bedrock_client passes aws_profile when configured."""

    def test_passes_aws_profile_when_configured(self):
        """When bedrock.profile is set, AnthropicBedrock receives aws_profile."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.AnthropicBedrock.return_value = mock_client

        with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
            with patch("agent.bedrock_adapter._load_bedrock_config",
                       return_value={"profile": "sandbox"}):
                from agent.anthropic_adapter import build_anthropic_bedrock_client
                result = build_anthropic_bedrock_client("us-west-2")

        call_kwargs = mock_sdk.AnthropicBedrock.call_args
        assert call_kwargs[1]["aws_profile"] == "sandbox"
        assert call_kwargs[1]["aws_region"] == "us-west-2"

    def test_no_aws_profile_when_not_configured(self):
        """When no bedrock.profile, AnthropicBedrock is called without aws_profile."""
        mock_sdk = MagicMock()
        mock_client = MagicMock()
        mock_sdk.AnthropicBedrock.return_value = mock_client

        with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
            with patch("agent.bedrock_adapter._load_bedrock_config",
                       return_value={"region": "us-east-1"}):
                from agent.anthropic_adapter import build_anthropic_bedrock_client
                result = build_anthropic_bedrock_client("us-east-1")

        call_kwargs = mock_sdk.AnthropicBedrock.call_args
        assert "aws_profile" not in call_kwargs[1]
        assert call_kwargs[1]["aws_region"] == "us-east-1"


class TestResolveBedrockRegionProfile:
    """resolve_bedrock_region respects bedrock.profile for profile-local regions."""

    def test_uses_configured_profile_for_botocore_region(self):
        from agent.bedrock_adapter import resolve_bedrock_region

        profile_session = MagicMock()
        profile_session.get_config_variable.return_value = "ap-northeast-1"

        with patch("botocore.session.Session", return_value=profile_session) as session_cls:
            with patch("agent.bedrock_adapter._load_bedrock_config",
                       return_value={"profile": "my-aws-profile"}):
                region = resolve_bedrock_region(env={})

        session_cls.assert_called_once_with(profile="my-aws-profile")
        assert region == "ap-northeast-1"

    def test_env_region_still_wins_over_profile(self):
        from agent.bedrock_adapter import resolve_bedrock_region

        with patch("agent.bedrock_adapter._load_bedrock_config") as load_config:
            region = resolve_bedrock_region(env={"AWS_REGION": "eu-west-1"})

        load_config.assert_not_called()
        assert region == "eu-west-1"
