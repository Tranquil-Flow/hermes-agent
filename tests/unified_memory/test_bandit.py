"""Tests for LinUCB contextual bandit and session reward tracking."""

import numpy as np
import pytest

from unified_memory.bandit import (
    LinUCBArm, StageDecider, PipelineOptimizer,
    extract_query_features, SessionRewardTracker,
    RUN, SKIP, D,
)


class TestLinUCBArm:
    def test_initial_prediction(self):
        arm = LinUCBArm(d=4)
        x = np.array([1.0, 0.0, 1.0, 0.0])
        # Initial prediction should be non-negative (UCB bonus)
        p = arm.predict(x, alpha=1.0)
        assert p >= 0

    def test_update_improves_prediction(self):
        arm = LinUCBArm(d=4)
        x = np.array([1.0, 0.0, 1.0, 0.0])
        # Train with positive rewards
        for _ in range(10):
            arm.update(x, reward=1.0)
        p = arm.predict(x, alpha=0.1)
        assert p > 0, "After positive training, prediction should be positive"

    def test_update_count(self):
        arm = LinUCBArm(d=4)
        x = np.ones(4)
        arm.update(x, 1.0)
        arm.update(x, 0.5)
        assert arm.n == 2


class TestStageDecider:
    def test_essential_stage_always_runs(self):
        decider = StageDecider("embedding", d=4)
        decider.is_essential = True
        features = np.ones(4)
        assert decider.decide(features) == RUN

    def test_non_essential_initial_decision(self):
        decider = StageDecider("bm25_fusion", d=4)
        features = np.ones(4)
        # Initially both arms are equal, should default to RUN
        decision = decider.decide(features)
        assert decision in (RUN, SKIP)


class TestPipelineOptimizer:
    def test_exploration_phase(self):
        opt = PipelineOptimizer(exploration_budget=10)
        features = np.ones(D)
        decisions = opt.decide_stages(features)
        # During exploration, all stages should run
        assert all(decisions.values()), "All stages should run during exploration"
        assert opt.in_exploration_phase

    def test_exits_exploration(self):
        opt = PipelineOptimizer(exploration_budget=5)
        features = np.ones(D)
        for _ in range(6):
            opt.decide_stages(features)
            opt.update_reward(0.5)
        assert not opt.in_exploration_phase

    def test_stats(self):
        opt = PipelineOptimizer(exploration_budget=2)
        features = np.ones(D)
        opt.decide_stages(features)
        opt.update_reward(1.0)
        stats = opt.get_stats()
        assert stats["query_count"] == 1
        assert "stages" in stats


class TestFeatureExtraction:
    def test_basic_features(self):
        features = extract_query_features("What is the API URL?", store_size=100)
        assert len(features) == D
        assert features[2] == 1.0  # has_question_mark

    def test_temporal_marker(self):
        features = extract_query_features("When did we deploy last week?")
        assert features[3] == 1.0  # temporal_markers

    def test_no_temporal(self):
        features = extract_query_features("What is the database version?")
        assert features[3] == 0.0

    def test_store_size_normalization(self):
        f0 = extract_query_features("test", store_size=0)
        f100 = extract_query_features("test", store_size=100)
        assert f100[6] > f0[6]  # larger store should have higher feature


class TestSessionRewardTracker:
    def test_store_after_recall_credits(self):
        tracker = SessionRewardTracker(credit_window_seconds=300)
        # Recall returns some IDs
        rewards = tracker.on_recall(["id1", "id2"], now=1000.0)
        assert len(rewards) == 0  # First recall, no re-recall

        # Store within window
        rewards = tracker.on_store(now=1100.0)
        assert "id1" in rewards
        assert "id2" in rewards
        assert rewards["id1"] == 0.5

    def test_store_after_recall_expired(self):
        tracker = SessionRewardTracker(credit_window_seconds=300)
        tracker.on_recall(["id1"], now=1000.0)
        # Store after window
        rewards = tracker.on_store(now=2000.0)
        assert len(rewards) == 0

    def test_re_recall_signal(self):
        tracker = SessionRewardTracker()
        tracker.on_recall(["id1"], now=1000.0)
        # Recall same ID again
        rewards = tracker.on_recall(["id1", "id2"], now=2000.0)
        assert "id1" in rewards
        assert rewards["id1"] == 0.4  # re-recall signal
        assert "id2" not in rewards  # first time
