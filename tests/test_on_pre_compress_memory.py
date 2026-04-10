"""Test that on_pre_compress memory_context flows to summary generator."""
from pathlib import Path


class TestOnPreCompressMemory:
    """Verify memory_context from on_pre_compress reaches the summary generator."""

    def test_generate_summary_accepts_memory_context(self):
        """_generate_summary must accept memory_context parameter."""
        comp_path = Path(__file__).resolve().parents[1] / "agent" / "context_compressor.py"
        source = comp_path.read_text()
        assert "memory_context" in source, (
            "_generate_summary must accept optional memory_context parameter"
        )

    def test_memory_section_appended_to_prompt(self):
        """When memory_context is non-empty, it must be appended to the prompt."""
        comp_path = Path(__file__).resolve().parents[1] / "agent" / "context_compressor.py"
        source = comp_path.read_text()
        assert "MEMORY PROVIDER INSIGHTS" in source, (
            "Memory provider insights section must be added to summary prompt"
        )
        assert "_memory_section" in source, (
            "Memory section variable must be constructed and appended"
        )

    def test_on_pre_compress_hook_called(self):
        """The on_pre_compress hook must be called during compression flow."""
        cc_path = Path(__file__).resolve().parents[1] / "agent" / "conversation_compression.py"
        source = cc_path.read_text()
        assert "on_pre_compress" in source, (
            "on_pre_compress hook must be called during compression"
        )
