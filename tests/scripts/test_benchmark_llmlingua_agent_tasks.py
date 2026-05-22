from scripts.benchmark_llmlingua_agent_tasks import (
    Fact,
    LLMLinguaConfig,
    LLMLinguaLocalCompressor,
    TaskCase,
    _prepare_compressor_for_benchmark,
    build_compressor,
    evaluate_text,
    run_task,
)


def test_evaluate_text_counts_required_and_optional_facts():
    task = TaskCase(
        id="demo",
        category="web",
        tool_name="web_extract",
        tool_args='{"urls": ["https://example.test"]}',
        question="What changed?",
        content="Version 2.3.1 fixes CVE-2026-4242. Upgrade by June 1.",
        facts=(
            Fact("version", r"2\.3\.1", required=True),
            Fact("cve", r"CVE-2026-4242", required=True),
            Fact("deadline", r"June 1", required=False),
        ),
    )

    result = evaluate_text(task, "Version 2.3.1 fixes CVE-2026-4242.")

    assert result["required_found"] == 2
    assert result["required_total"] == 2
    assert result["optional_found"] == 0
    assert result["answerable"] is True
    assert result["fact_recall"] == 2 / 3


def test_evaluate_text_marks_missing_required_fact_unanswerable():
    task = TaskCase(
        id="demo",
        category="web",
        tool_name="web_extract",
        tool_args='{}',
        question="What changed?",
        content="Version 2.3.1 fixes CVE-2026-4242.",
        facts=(
            Fact("version", r"2\.3\.1", required=True),
            Fact("cve", r"CVE-2026-4242", required=True),
        ),
    )

    result = evaluate_text(task, "Version 2.3.1 is mentioned.")

    assert result["required_found"] == 1
    assert result["answerable"] is False
    assert result["missing_required"] == ["cve"]


def test_drop_mode_uses_historical_summary_and_loses_required_facts():
    task = TaskCase(
        id="drop-demo",
        category="web",
        tool_name="web_extract",
        tool_args='{"urls": ["https://example.test/security"]}',
        question="Which CVE was fixed?",
        content="Security advisory: Version 2.3.1 fixes CVE-2026-4242." * 100,
        facts=(Fact("cve", r"CVE-2026-4242", required=True),),
    )

    result = run_task(task, build_compressor("drop"))

    assert result["mode"] == "drop"
    assert result["answerable"] is False
    assert "[web_extract]" in result["compressed_preview"]


def test_full_mode_is_oracle_and_preserves_required_facts():
    task = TaskCase(
        id="full-demo",
        category="web",
        tool_name="web_extract",
        tool_args='{}',
        question="Which CVE was fixed?",
        content="Security advisory: Version 2.3.1 fixes CVE-2026-4242." * 5,
        facts=(Fact("cve", r"CVE-2026-4242", required=True),),
    )

    result = run_task(task, build_compressor("full"))

    assert result["mode"] == "full"
    assert result["answerable"] is True
    assert result["compressed_chars"] == result["input_chars"]
    assert result["compression_ratio"] == 1.0


def test_benchmark_prepares_llmlingua_synchronously():
    class RecordingLLMLinguaCompressor(LLMLinguaLocalCompressor):
        def __init__(self):
            super().__init__(LLMLinguaConfig())
            self.loaded_for_benchmark = False

        def _ensure_loaded(self) -> None:
            self.loaded_for_benchmark = True

    compressor = RecordingLLMLinguaCompressor()

    _prepare_compressor_for_benchmark(compressor)

    assert compressor.loaded_for_benchmark is True
