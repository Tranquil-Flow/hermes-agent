"""Conversation ingestion strategies for cognitive memory.

Converts multi-turn conversations into meaningful memory entries.
Three strategies:
  - raw: store each turn as-is (current default, baseline)
  - chunk: group consecutive turns into semantic chunks
  - summarize: LLM-summarize sessions into key facts

The chunking strategy detects topic shifts via token overlap between
consecutive turns and groups related turns together.
"""
import logging
import re
from typing import List, Dict, Optional, Callable, Tuple

logger = logging.getLogger(__name__)

# Stop words for overlap computation
_STOP_WORDS = frozenset({
    'the','a','an','is','are','was','were','with','and','or','for',
    'in','on','at','to','of','it','its','by','as','that','this',
    'from','has','have','be','been','what','how','where','when',
    'which','who','do','does','did','not','but','so','if','than',
    'then','about','up','out','i','we','you','he','she','they',
    'me','us','my','our','your','just','also','would','could',
    'should','can','will','yeah','yes','no','oh','well','like',
    'really','right','okay','ok','sure','think','know','got',
    'going','want','need','say','said','tell','told','get',
    'make','take','come','go','see','look','good','great',
    'much','very','too','some','any','all','been','being',
    'had','having','did','doing','here','there',
})

def _tokenize(text: str) -> set:
    """Extract significant tokens from text."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}


def _token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity between significant tokens of two texts."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def ingest_raw(
    turns: List[Dict[str, str]],
    store,
    importance_fn: Optional[Callable] = None,
    time_gap: float = 0.0001,
) -> int:
    """Store each turn as a separate memory (baseline strategy).
    
    Args:
        turns: List of {"role": "user"/"assistant", "content": "..."}
        store: BenchmarkableStore with store() and simulate_time()
        importance_fn: Optional (role, content) -> float
        time_gap: Days to simulate between turns
    Returns:
        Number of memories stored
    """
    count = 0
    for turn in turns:
        content = turn.get("content", "").strip()
        if not content:
            continue
        role = turn.get("role", "user")
        importance = importance_fn(role, content) if importance_fn else (0.5 if role == "user" else 0.3)
        store.store(content, category="factual", importance=importance)
        count += 1
        if time_gap > 0:
            store.simulate_time(time_gap)
    return count


def ingest_chunked(
    turns: List[Dict[str, str]],
    store,
    chunk_size: int = 4,
    overlap_threshold: float = 0.03,
    max_chunk_chars: int = 6000,
    importance_fn: Optional[Callable] = None,
    time_gap: float = 0.01,
) -> int:
    """Group consecutive turns into semantic chunks before storing.
    
    Chunking strategy:
    1. Start a new chunk
    2. Add turns until:
       a) chunk_size turns reached, OR
       b) token overlap with previous turn drops below threshold (topic shift), OR
       c) max_chunk_chars reached
    3. Store the chunk as a single memory with all turns joined
    4. Start next chunk from the topic-shift turn
    
    This preserves conversational context while reducing memory count
    and improving semantic coherence of each stored unit.
    """
    if not turns:
        return 0
    
    chunks = []
    current_chunk = []
    current_chars = 0
    prev_content = ""
    
    for turn in turns:
        content = turn.get("content", "").strip()
        if not content:
            continue
        
        role = turn.get("role", "user")
        speaker = turn.get("speaker", role)
        
        # Check for topic shift
        should_break = False
        if current_chunk:
            if len(current_chunk) >= chunk_size:
                should_break = True
            elif current_chars + len(content) > max_chunk_chars:
                should_break = True
            elif prev_content and _token_overlap(prev_content, content) < overlap_threshold:
                # Topic shift detected — but only if we have at least 2 turns
                if len(current_chunk) >= 2:
                    should_break = True
        
        if should_break and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        
        current_chunk.append({"speaker": speaker, "content": content, "role": role})
        current_chars += len(content)
        prev_content = content
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Store each chunk as a single memory
    count = 0
    for chunk in chunks:
        # Format: "Speaker1: text\nSpeaker2: text\n..."
        lines = []
        for turn in chunk:
            speaker = turn["speaker"]
            lines.append(f"{speaker}: {turn['content']}")
        chunk_text = "\n".join(lines)
        
        # Importance: average across turns, slightly higher for user-heavy chunks
        if importance_fn:
            importances = [importance_fn(t["role"], t["content"]) for t in chunk]
            importance = sum(importances) / len(importances)
        else:
            user_ratio = sum(1 for t in chunk if t["role"] == "user") / len(chunk)
            importance = 0.4 + 0.2 * user_ratio
        
        store.store(chunk_text, category="factual", importance=importance)
        count += 1
        if time_gap > 0:
            store.simulate_time(time_gap)
    
    logger.info("Chunked %d turns into %d chunks (avg %.1f turns/chunk)",
                sum(len(c) for c in chunks), len(chunks),
                sum(len(c) for c in chunks) / max(len(chunks), 1))
    return count


def ingest_summarized(
    turns: List[Dict[str, str]],
    store,
    llm_fn: Callable[[str], str],
    max_turns_per_summary: int = 20,
    importance: float = 0.7,
    time_gap: float = 0.1,
) -> int:
    """LLM-summarize conversation segments into key facts before storing.
    
    Args:
        turns: List of conversation turns
        store: BenchmarkableStore
        llm_fn: Callable(conversation_text) -> summary_text
        max_turns_per_summary: Split long conversations into segments
        importance: Importance for summarized facts (higher = more trusted)
        time_gap: Days between summary memories
    """
    if not turns or not llm_fn:
        return ingest_raw(turns, store)
    
    # Split into segments
    segments = []
    for i in range(0, len(turns), max_turns_per_summary):
        segment = turns[i:i + max_turns_per_summary]
        segments.append(segment)
    
    count = 0
    for segment in segments:
        # Build conversation text
        lines = []
        for turn in segment:
            role = turn.get("speaker", turn.get("role", "user"))
            content = turn.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        conv_text = "\n".join(lines)
        
        # Summarize via LLM
        prompt = (
            "Extract the key facts, preferences, decisions, and events from this "
            "conversation. Return each fact on its own line. Be specific and include "
            "names, dates, numbers, and concrete details. Skip greetings and filler.\n\n"
            f"{conv_text}\n\nKey facts:"
        )
        
        try:
            summary = llm_fn(prompt)
        except Exception as e:
            logger.warning("LLM summarization failed: %s. Falling back to chunked.", e)
            count += ingest_chunked(segment, store)
            continue
        
        # Store each fact line as a separate memory
        facts = [line.strip().lstrip('- \u2022*') for line in summary.strip().split('\n')]
        facts = [f for f in facts if f and len(f) > 10]
        
        for fact in facts:
            store.store(fact, category="factual", importance=importance)
            count += 1
        
        if time_gap > 0:
            store.simulate_time(time_gap)
    
    logger.info("Summarized %d turns into %d facts across %d segments",
                len(turns), count, len(segments))
    return count
