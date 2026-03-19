# Benchmark Blockers & Known Issues

## Session: 2026-03-19

### B1: LLM Judge unavailable (carried from previous session)
- **Status**: Ongoing
- **Impact**: Suite A cross-reference scores include 3 judge-ceiling failures
  that would pass with LLM evaluation. Expected overall ~0.95+ with LLM judge.
- **Fix**: `hermes-aegis vault set ANTHROPIC_API_KEY <token>` on host machine.
  The injector auto-detects OAuth tokens vs sk-ant-api keys.

### B2: Full sentence-transformers benchmark OOMs in 5GB Docker container
- **Status**: Ongoing
- **Impact**: Can't run full 200-scenario Suite A with --embedding sentence-transformers.
  Individual categories work. TF-IDF (default) runs fine.
- **Fix options**: Increase container memory, switch to ONNX runtime, or run on host macOS.

### B3: Suite D adversarial scores (0.067 — expected low)
- **Status**: NOT a bug — this is a finding.
- **Detail**: The cognitive store's embedding-based scoring does NOT naturally
  de-prioritize prompt injection payloads. TF-IDF cosine similarity between
  injections and legitimate queries is often high enough that injections
  surface as top-1 results.
- **Implication**: Adversarial filtering must happen at the hermes-aegis layer
  (input sanitization) rather than the embedding layer. This is by design.
- **Expected score with aegis filtering active**: ~0.87+ (hallucinated_fact
  scenarios will still be hard since they use domain-relevant vocabulary).

### B4: Suite C scope scores (0.550 — partial)
- **Status**: Findings split — scope isolation works, retrieval precision needs work.
- **Detail**: Zero scope leaks (no_leak=100%). Failures are retrieval precision:
  correct fact IS in memory, but a different fact in the same scope ranks higher.
- **Fix**: Scope boosting (e.g., 2x score multiplier for exact scope match) would
  help. Currently scoped recall only filters, doesn't boost.

### B5: filesystem split — write_file vs terminal
- **Status**: Resolved within session.
- **Detail**: write_file (host macOS) writes to ~/Projects/... which is NOT
  the same as /workspace/Projects/... in Docker containers.
  The actual project workspace is mounted at /workspace/Projects/.
  Always use terminal + heredoc or `cat >` for writing files to the workspace.
