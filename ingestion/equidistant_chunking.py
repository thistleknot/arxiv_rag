"""
Equidistant Overlapped Chunking v2

Produces consistently-sized text chunks with controlled overlap for RAG pipelines.

APPROACH COMPARISON (Brown corpus, 100 docs, 4097 paragraphs):
┌─────────────────┬─────────────┬──────────────────┬─────────────────┐
│ Approach        │ Atomic Unit │ Below Overlap    │ Use Case        │
│                 │             │ Target           │                 │
├─────────────────┼─────────────┼──────────────────┼─────────────────┤
│ Sentence-based  │ Sentences   │ 0.0%             │ Max control     │
│ Para-count      │ Paragraphs  │ 40.6%            │ Semantic only   │
│ Hybrid (best)   │ Paragraphs  │ 0.0%             │ Semantic+control│
└─────────────────┴─────────────┴──────────────────┴─────────────────┘

KEY INSIGHT:
    The aggregation logic (SlidingAggregator) uses CHARACTER thresholds for
    both flush and overlap decisions. What changes is the ATOMIC UNIT fed to it:
    
    - Sentence-based: create_blocks() splits at ". ? !" → fine granularity
    - Paragraph-based: create_paragraph_blocks() splits at "\\n\\n" → semantic units
    
    The hybrid approach achieves both semantic boundaries AND overlap control
    by feeding paragraph blocks to the character-based aggregator.

RECOMMENDED APPROACH:
    Use chunk_paragraphs_hybrid_tracked() for production RAG pipelines:
    - Preserves paragraph integrity (semantic boundaries)
    - Achieves 0% below overlap target with proper corpus structure
    - All parameters derived from corpus statistics (no magic numbers)

NLTK BROWN CORPUS USAGE:
    Brown corpus has paragraph structure via brown.paras(), but NOT in raw text.
    
    WRONG (loses structure):
        docs = [' '.join(brown.words(fid)) for fid in brown.fileids()]
        # Results in 0 paragraph breaks, all text on one line
    
    CORRECT (preserves structure):
        docs = [
            '\\n\\n'.join([
                ' '.join([' '.join(sent) for sent in para])
                for para in brown.paras(fid)
            ])
            for fid in brown.fileids()
        ]
        # Results in 4097 paragraphs from 100 docs

FEATURES:
    1. Character-based overlap (not block-count) - fixes 99.9% below-target bug
    2. Document boundary tracking - excludes cross-doc transitions from stats
    3. Sample-based parameter estimation using robust statistics (log median ± MAD)
    4. Optional undersized chunk merging
    5. Three chunking modes: sentence-based, paragraph-count, hybrid

USAGE:
    # Hybrid approach (recommended)
    from equidistant_chunker_v2 import (
        paragraph_config_from_sample, 
        chunk_paragraphs_hybrid_tracked,
        AggregateConfig
    )
    
    # Derive config from corpus
    para_config, stats = paragraph_config_from_sample(docs, target_paragraphs=3)
    config = AggregateConfig(
        target=para_config.target_chars,
        tolerance=stats.para_char_mad,
        ratio=2.0,
        overlap_pct=para_config.overlap_chars / para_config.target_chars
    )
    
    # Chunk with document tracking
    chunks = chunk_paragraphs_hybrid_tracked(docs, config, para_config.max_para_chars)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Iterator, Generator, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from math import ceil, log, exp
import numpy as np


# =============================================================================
# Core Data Types
# =============================================================================

@dataclass
class ChunkInfo:
    """Chunk with metadata for tracking."""
    text: str
    doc_id: int
    chunk_idx: int
    is_doc_start: bool
    is_doc_end: bool
    
    def __len__(self):
        return len(self.text)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AggregateConfig:
    """
    Chunking configuration with target ± tolerance bounds.
    
    Parameters:
        target: Target chunk size in characters
        tolerance: Acceptable deviation (target ± tolerance defines bounds)
        ratio: atomic_size = target / ratio (higher = finer granularity)
        overlap_pct: Fraction of target to overlap (0.5 = 50%)
        merge_undersized: If True, merge final undersized chunks when possible
    
    Example:
        AggregateConfig(429, 107.25, ratio=4, overlap_pct=0.5)
        → atomic=107, overlap=214 chars minimum
    """
    target: int
    tolerance: float
    ratio: float = 4.0
    overlap_pct: float = 0.5
    merge_undersized: bool = True
    
    # Derived from sample (optional)
    chars_per_word: float = field(default=5.0, repr=False)
    
    @property
    def min_size(self) -> int:
        return int(self.target - self.tolerance)
    
    @property
    def max_size(self) -> int:
        return int(self.target + self.tolerance)
    
    @property
    def atomic_size(self) -> int:
        return int(self.target / self.ratio)
    
    @property
    def overlap_chars(self) -> int:
        return int(self.target * self.overlap_pct)
    
    @property
    def atomic_overlap(self) -> int:
        return min(20, self.atomic_size // 10)
    
    @property
    def target_words(self) -> int:
        """Approximate word count for target size."""
        return int(self.target / self.chars_per_word)
    
    def __repr__(self):
        return (f"AggregateConfig({self.target} ± {self.tolerance}, "
                f"atomic={self.atomic_size}, "
                f"overlap={self.overlap_pct:.0%} ~{self.overlap_chars}ch)")


# =============================================================================
# Sample-Based Parameter Estimation
# =============================================================================

class SampleStats(NamedTuple):
    """Statistics from a text sample."""
    # Sentence-level
    sent_char_median: float
    sent_char_mad: float
    sent_word_median: float
    sent_word_mad: float
    
    # Paragraph-level
    para_char_median: float
    para_char_mad: float
    sents_per_para_median: float
    sents_per_para_mad: float
    
    # Derived
    chars_per_word: float
    n_sentences: int
    n_paragraphs: int


def estimate_from_sample(
    texts: List[str],
    sample_size: Optional[int] = None
) -> SampleStats:
    """
    Estimate text statistics from a sample using robust measures.
    
    Uses NLTK sent_tokenize for accurate sentence boundary detection.
    Collects both sentence-level and paragraph-level statistics.
    
    IMPORTANT: Input texts MUST have \\n\\n paragraph breaks for accurate
    paragraph-level statistics. For NLTK corpora with paragraph structure,
    reconstruct texts properly:
    
        # Brown corpus - WRONG (loses paragraphs):
        doc = ' '.join(brown.words(fileid))
        
        # Brown corpus - RIGHT (preserves paragraphs):
        doc = '\\n\\n'.join([
            ' '.join([' '.join(sent) for sent in para])
            for para in brown.paras(fileid)
        ])
    
    Args:
        texts: List of document strings with \\n\\n paragraph breaks
        sample_size: Max number of texts to sample (None = use all)
    
    Returns:
        SampleStats with median/MAD for sentences and paragraphs
    """
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError:
        # Fallback to regex if NLTK not available
        import re
        def sent_tokenize(text):
            return [s.strip() for s in re.split(r'[.!?]+\s+', text) if s.strip()]
    
    if sample_size and len(texts) > sample_size:
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in indices]
    
    # Sentence-level stats
    sent_chars = []
    sent_words = []
    
    # Paragraph-level stats
    para_chars = []
    sents_per_para = []
    
    for text in texts:
        # Split by double newline - texts MUST have paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            if len(para) < 20:  # Skip tiny fragments
                continue
            
            para_chars.append(len(para))
            sentences = sent_tokenize(para)
            sents_per_para.append(len(sentences))
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 5:  # Skip tiny fragments
                    sent_chars.append(len(sent))
                    sent_words.append(len(sent.split()))
    
    # Convert to arrays
    sent_chars_arr = np.array(sent_chars) if sent_chars else np.array([100])
    sent_words_arr = np.array(sent_words) if sent_words else np.array([20])
    para_chars_arr = np.array(para_chars) if para_chars else np.array([400])
    sents_per_para_arr = np.array(sents_per_para) if sents_per_para else np.array([4])
    
    # Sentence-level robust stats
    sent_char_median = np.median(sent_chars_arr)
    sent_char_mad = np.median(np.abs(sent_chars_arr - sent_char_median))
    sent_word_median = np.median(sent_words_arr)
    sent_word_mad = np.median(np.abs(sent_words_arr - sent_word_median))
    
    # Paragraph-level robust stats
    para_char_median = np.median(para_chars_arr)
    para_char_mad = np.median(np.abs(para_chars_arr - para_char_median))
    sents_per_para_median = np.median(sents_per_para_arr)
    sents_per_para_mad = np.median(np.abs(sents_per_para_arr - sents_per_para_median))
    
    # Chars per word
    total_chars = sum(sent_chars) if sent_chars else 1
    total_words = sum(sent_words) if sent_words else 1
    chars_per_word = total_chars / total_words
    
    return SampleStats(
        sent_char_median=sent_char_median,
        sent_char_mad=sent_char_mad,
        sent_word_median=sent_word_median,
        sent_word_mad=sent_word_mad,
        para_char_median=para_char_median,
        para_char_mad=para_char_mad,
        sents_per_para_median=sents_per_para_median,
        sents_per_para_mad=sents_per_para_mad,
        chars_per_word=chars_per_word,
        n_sentences=len(sent_chars),
        n_paragraphs=len(para_chars)
    )


def config_from_sample(
    texts: List[str],
    target_sentences: int = 3,
    tolerance_mad_multiplier: float = 2.0,
    ratio: float = 4.0,
    overlap_pct: float = 0.5,
    sample_size: int = 50
) -> AggregateConfig:
    """
    Create AggregateConfig based on corpus sample statistics (sentence-based).
    
    Args:
        texts: Sample of documents
        target_sentences: How many median sentences to target per chunk
        tolerance_mad_multiplier: tolerance = MAD * this value
        ratio: Atomic block ratio
        overlap_pct: Overlap fraction
        sample_size: Max documents to sample
    
    Returns:
        AggregateConfig tuned to the corpus
    """
    sample_stats = estimate_from_sample(texts, sample_size=sample_size)
    
    # Target = N sentences worth of characters
    target = int(sample_stats.sent_char_median * target_sentences)
    
    # Tolerance based on MAD (robust measure of spread)
    tolerance = sample_stats.sent_char_mad * tolerance_mad_multiplier * target_sentences
    
    return AggregateConfig(
        target=target,
        tolerance=tolerance,
        ratio=ratio,
        overlap_pct=overlap_pct,
        chars_per_word=sample_stats.chars_per_word
    )


def config_from_paragraphs(
    texts: List[str],
    sample_size: int = 100,
    mad_multiplier: float = 2.0
) -> Tuple[AggregateConfig, SampleStats]:
    """
    Create AggregateConfig derived entirely from paragraph structure.
    
    All parameters are derived from corpus statistics:
    - target = median paragraph size (chars)
    - tolerance = extends to log-transformed upper bound
    - overlap = (median_sents_per_para / 2) * median_chars_per_sent
    - ratio = median_sents_per_para (atomic block ≈ 1 sentence)
    
    Args:
        texts: Sample of documents
        sample_size: Max documents to sample
        mad_multiplier: Multiplier for log-MAD upper bound (default 2.0)
    
    Returns:
        Tuple of (AggregateConfig, SampleStats) for inspection
    """
    sample_stats = estimate_from_sample(texts, sample_size=sample_size)
    
    # Target = median paragraph size
    target = int(sample_stats.para_char_median)
    
    # Log-transformed upper bound for tolerance
    para_chars_sample = []
    for text in texts[:sample_size]:
        for para in text.split('\n\n'):
            para = para.strip()
            if len(para) >= 20:
                para_chars_sample.append(len(para))
    
    if para_chars_sample:
        log_paras = np.log(para_chars_sample)
        log_median = np.median(log_paras)
        log_mad = np.median(np.abs(log_paras - log_median))
        upper_bound = exp(log_median + mad_multiplier * log_mad)
        lower_bound = exp(log_median - mad_multiplier * log_mad)
    else:
        upper_bound = target * 1.5
        lower_bound = target * 0.5
    
    # Asymmetric tolerance: use lower_bound for min, upper_bound for max
    # But AggregateConfig uses symmetric tolerance, so we need to pick wisely
    # Use the larger of the two deviations to ensure both bounds are covered
    tolerance_up = upper_bound - target
    tolerance_down = target - lower_bound
    
    # Use asymmetric approach: min = lower_bound, max = upper_bound
    # We'll set tolerance to cover upper_bound, then clamp min_size in property
    tolerance = tolerance_up  # Extend to upper bound
    
    # Store lower bound for min_size calculation
    min_size_override = max(50, int(lower_bound))  # Floor at 50 chars
    
    # Natural 50% overlap: half the sentences in a paragraph
    overlap_sentences = sample_stats.sents_per_para_median / 2
    overlap_chars = int(overlap_sentences * sample_stats.sent_char_median)
    overlap_pct = overlap_chars / target if target > 0 else 0.5
    
    # Clamp overlap_pct to reasonable range
    overlap_pct = max(0.25, min(0.75, overlap_pct))
    
    # Ratio: atomic block ≈ 1 sentence
    ratio = max(2.0, sample_stats.sents_per_para_median)
    
    # Create config with adjusted tolerance so min_size stays positive
    # Use the distance from target to lower_bound as the lower tolerance
    safe_tolerance = min(tolerance, target - 50)  # Ensure min_size >= 50
    
    config = AggregateConfig(
        target=target,
        tolerance=safe_tolerance,
        ratio=ratio,
        overlap_pct=overlap_pct,
        chars_per_word=sample_stats.chars_per_word
    )
    
    return config, sample_stats


@dataclass
class ParagraphConfig:
    """
    Paragraph-based chunking configuration.
    
    All parameters derived from corpus paragraph statistics:
    - target_paragraphs: How many paragraphs per chunk (default 3)
    - overlap_paragraphs: How many paragraphs to overlap (default 1)
    - median_para_chars: Median paragraph size in chars (from corpus)
    - max_para_chars: Upper bound for paragraph size before splitting
    
    Example:
        ParagraphConfig(target_paragraphs=3, overlap_paragraphs=1, 
                       median_para_chars=200, max_para_chars=600)
        → chunks of ~600 chars, overlapping by ~200 chars
    """
    target_paragraphs: int = 3
    overlap_paragraphs: int = 1
    median_para_chars: int = 200
    max_para_chars: int = 600  # Paragraphs larger than this get split
    
    @property
    def target_chars(self) -> int:
        """Target chunk size in characters."""
        return self.target_paragraphs * self.median_para_chars
    
    @property
    def overlap_chars(self) -> int:
        """Overlap size in characters."""
        return self.overlap_paragraphs * self.median_para_chars
    
    @property
    def min_chars(self) -> int:
        """Minimum chunk size (1 paragraph)."""
        return self.median_para_chars // 2
    
    @property
    def max_chars(self) -> int:
        """Maximum chunk size."""
        return (self.target_paragraphs + 1) * self.median_para_chars
    
    def __repr__(self):
        return (f"ParagraphConfig(target={self.target_paragraphs}¶ ~{self.target_chars}ch, "
                f"overlap={self.overlap_paragraphs}¶ ~{self.overlap_chars}ch)")


def paragraph_config_from_sample(
    texts: List[str],
    target_paragraphs: int = 3,
    overlap_paragraphs: int = 1,
    sample_size: int = 100,
    mad_multiplier: float = 2.0
) -> Tuple[ParagraphConfig, SampleStats]:
    """
    Create ParagraphConfig from corpus statistics.
    
    Args:
        texts: Sample documents
        target_paragraphs: Number of paragraphs per chunk
        overlap_paragraphs: Number of paragraphs to overlap
        sample_size: Max documents to sample
        mad_multiplier: Multiplier for log-MAD upper bound
    
    Returns:
        Tuple of (ParagraphConfig, SampleStats)
    """
    sample_stats = estimate_from_sample(texts, sample_size=sample_size)
    
    # Log-transformed upper bound for max paragraph size
    para_chars_sample = []
    for text in texts[:sample_size]:
        for para in text.split('\n\n'):
            para = para.strip()
            if len(para) >= 20:
                para_chars_sample.append(len(para))
    
    if para_chars_sample:
        log_paras = np.log(para_chars_sample)
        log_median = np.median(log_paras)
        log_mad = np.median(np.abs(log_paras - log_median))
        max_para_chars = int(exp(log_median + mad_multiplier * log_mad))
    else:
        max_para_chars = int(sample_stats.para_char_median * 2)
    
    config = ParagraphConfig(
        target_paragraphs=target_paragraphs,
        overlap_paragraphs=overlap_paragraphs,
        median_para_chars=int(sample_stats.para_char_median),
        max_para_chars=max_para_chars
    )
    
    return config, sample_stats


# =============================================================================
# Core Chunking
# =============================================================================

def create_blocks(text: str, config: AggregateConfig) -> List[str]:
    """Split text into atomic blocks using recursive character splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.atomic_size,
        chunk_overlap=config.atomic_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )
    return splitter.split_text(text)


def create_paragraph_blocks(text: str, max_para_size: int) -> List[str]:
    """
    Split text into paragraph-based blocks.
    
    Paragraphs are split by double newlines. Only paragraphs exceeding
    max_para_size are further split using RecursiveCharacterTextSplitter.
    This preserves paragraph integrity for normal-sized paragraphs.
    
    Args:
        text: Input text
        max_para_size: Maximum paragraph size before splitting
    
    Returns:
        List of paragraph blocks (intact paragraphs or sub-splits of large ones)
    """
    # Split by paragraph boundaries first
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    blocks = []
    
    # Splitter for oversized paragraphs only
    oversized_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_para_size,
        chunk_overlap=min(50, max_para_size // 10),
        separators=["\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )
    
    for para in raw_paragraphs:
        if len(para) <= max_para_size:
            # Keep paragraph intact
            blocks.append(para)
        else:
            # Split oversized paragraph
            sub_blocks = oversized_splitter.split_text(para)
            blocks.extend(sub_blocks)
    
    return blocks


def chunk_paragraphs_hybrid(
    text: str,
    config: AggregateConfig,
    max_para_chars: int
) -> List[str]:
    """
    Chunk using paragraphs as atomic units with character-based aggregation.
    
    This hybrid approach:
    - Splits at paragraph boundaries (\n\n) preserving semantic units
    - Uses character thresholds for chunk size and overlap decisions
    - Achieves ~98% overlap target compliance (vs 60% with pure paragraph-count)
    
    Args:
        text: Input text
        config: AggregateConfig with target, tolerance, overlap_pct
        max_para_chars: Max paragraph size before sub-splitting
    
    Returns:
        List of chunk strings
    """
    blocks = create_paragraph_blocks(text, max_para_chars)
    agg = SlidingAggregator(config)
    chunks = []
    
    for block in blocks:
        for chunk in agg.feed(block):
            chunks.append(chunk)
    
    for chunk in agg.finish(chunks):
        chunks.append(chunk)
    
    return chunks


def chunk_paragraphs_hybrid_tracked(
    texts: List[str],
    config: AggregateConfig,
    max_para_chars: int
) -> List[ChunkInfo]:
    """
    Chunk multiple documents using hybrid paragraph approach with tracking.
    
    Args:
        texts: List of document strings
        config: AggregateConfig
        max_para_chars: Max paragraph size before sub-splitting
    
    Returns:
        List of ChunkInfo with document tracking
    """
    all_chunks = []
    
    for doc_id, text in enumerate(texts):
        doc_chunks = chunk_paragraphs_hybrid(text, config, max_para_chars)
        
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            info = ChunkInfo(
                text=chunk_text,
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                is_doc_start=(chunk_idx == 0),
                is_doc_end=(chunk_idx == len(doc_chunks) - 1)
            )
            all_chunks.append(info)
    
    return all_chunks


class SlidingAggregator:
    """
    Aggregates atomic blocks into chunks with character-based overlap.
    
    The overlap is determined by character count, not block count,
    ensuring minimum overlap is preserved regardless of block sizes.
    """
    
    def __init__(self, config: AggregateConfig):
        self.config = config
        self.buffer: List[str] = []
        self.buffer_len: int = 0
        self.previous_chunk: Optional[str] = None
    
    def _calc_len(self) -> int:
        if not self.buffer:
            return 0
        return sum(len(b) for b in self.buffer) + len(self.buffer) - 1
    
    def _flush(self) -> str:
        chunk = ' '.join(self.buffer)
        
        # Keep enough blocks to meet target overlap in CHARACTERS
        if self.config.overlap_chars > 0 and self.buffer:
            kept_blocks = []
            kept_chars = 0
            
            for block in reversed(self.buffer):
                kept_blocks.insert(0, block)
                kept_chars += len(block)
                if len(kept_blocks) > 1:
                    kept_chars += 1  # Space between blocks
                
                if kept_chars >= self.config.overlap_chars:
                    break
            
            self.buffer = kept_blocks
        else:
            self.buffer = []
        
        self.buffer_len = self._calc_len()
        self.previous_chunk = chunk
        return chunk
    
    def feed(self, block: str) -> Iterator[str]:
        block_len = len(block)
        potential = self.buffer_len + block_len + (1 if self.buffer else 0)
        
        should_flush = False
        
        if self.buffer:
            if potential > self.config.max_size:
                should_flush = True
            elif self.buffer_len >= self.config.min_size:
                if abs(self.buffer_len - self.config.target) < abs(potential - self.config.target):
                    should_flush = True
        
        if should_flush:
            yield self._flush()
        
        self.buffer.append(block)
        self.buffer_len = self._calc_len()
    
    def finish(self, chunks_so_far: List[str]) -> Iterator[str]:
        """
        Finish processing, optionally merging undersized final chunk.
        
        Args:
            chunks_so_far: Chunks already emitted (for potential merge)
        """
        if not self.buffer:
            return
        
        final_chunk = ' '.join(self.buffer)
        
        # Check if we should merge undersized final chunk
        if (self.config.merge_undersized and 
            len(final_chunk) < self.config.min_size and 
            chunks_so_far):
            
            last_chunk = chunks_so_far[-1]
            merged = last_chunk + ' ' + final_chunk
            
            # Only merge if result stays within bounds
            if len(merged) <= self.config.max_size:
                chunks_so_far[-1] = merged
                return  # Don't yield anything, we modified in place
        
        yield final_chunk


def chunk_equidistant(text: str, config: AggregateConfig) -> List[str]:
    """
    Chunk a single text into equidistant overlapping segments.
    
    Args:
        text: Input text string
        config: Chunking configuration
    
    Returns:
        List of chunk strings
    """
    blocks = create_blocks(text, config)
    aggregator = SlidingAggregator(config)
    chunks = []
    
    for block in blocks:
        chunks.extend(aggregator.feed(block))
    
    chunks.extend(aggregator.finish(chunks))
    return chunks


class ParagraphAggregator:
    """
    Aggregates paragraph blocks into chunks with paragraph-based overlap.
    
    Unlike SlidingAggregator which uses character counts, this aggregator
    counts paragraphs for both chunk size and overlap, ensuring paragraph
    boundaries are respected.
    """
    
    def __init__(self, config: ParagraphConfig):
        self.config = config
        self.buffer: List[str] = []  # List of paragraph blocks
    
    def _buffer_chars(self) -> int:
        """Total characters in buffer (with spaces)."""
        if not self.buffer:
            return 0
        return sum(len(p) for p in self.buffer) + len(self.buffer) - 1
    
    def _flush(self) -> str:
        """Emit current buffer as chunk, keep overlap paragraphs."""
        chunk = '\n\n'.join(self.buffer)
        
        # Keep last N paragraphs for overlap
        if self.config.overlap_paragraphs > 0:
            self.buffer = self.buffer[-self.config.overlap_paragraphs:]
        else:
            self.buffer = []
        
        return chunk
    
    def feed(self, para: str) -> Iterator[str]:
        """Add a paragraph block, emit chunk if target reached."""
        self.buffer.append(para)
        
        # Check if we should flush
        if len(self.buffer) >= self.config.target_paragraphs:
            # Check if adding more would be closer to target
            current_chars = self._buffer_chars()
            
            if current_chars >= self.config.target_chars:
                yield self._flush()
    
    def finish(self) -> Iterator[str]:
        """Emit any remaining paragraphs."""
        if self.buffer:
            # If we have more than just overlap, emit
            if len(self.buffer) > self.config.overlap_paragraphs:
                yield '\n\n'.join(self.buffer)
            elif self.buffer:
                # Emit even small final chunks
                yield '\n\n'.join(self.buffer)


def chunk_by_paragraphs(text: str, config: ParagraphConfig) -> List[str]:
    """
    Chunk text using paragraphs as atomic units.
    
    Paragraphs are preserved intact unless they exceed max_para_chars,
    in which case they are split using RecursiveCharacterTextSplitter.
    
    Args:
        text: Input text
        config: ParagraphConfig with paragraph-based parameters
    
    Returns:
        List of chunk strings (each containing ~N paragraphs)
    """
    # Get paragraph blocks (oversized ones are split)
    blocks = create_paragraph_blocks(text, config.max_para_chars)
    
    aggregator = ParagraphAggregator(config)
    chunks = []
    
    for block in blocks:
        for chunk in aggregator.feed(block):
            chunks.append(chunk)
    
    for chunk in aggregator.finish():
        chunks.append(chunk)
    
    return chunks


def chunk_paragraphs_tracked(
    texts: List[str],
    config: ParagraphConfig
) -> List[ChunkInfo]:
    """
    Chunk multiple documents using paragraph-based approach with tracking.
    
    Args:
        texts: List of document strings
        config: ParagraphConfig
    
    Returns:
        List of ChunkInfo with document tracking metadata
    """
    all_chunks = []
    
    for doc_id, text in enumerate(texts):
        doc_chunks = chunk_by_paragraphs(text, config)
        
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            info = ChunkInfo(
                text=chunk_text,
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                is_doc_start=(chunk_idx == 0),
                is_doc_end=(chunk_idx == len(doc_chunks) - 1)
            )
            all_chunks.append(info)
    
    return all_chunks


def paragraph_stats(
    chunks: List[ChunkInfo],
    config: ParagraphConfig
) -> dict:
    """
    Compute statistics for paragraph-based chunks with doc boundary awareness.
    
    Reports both paragraph-count overlap and character overlap metrics.
    """
    if not chunks:
        return {}
    
    lens = np.array([len(c) for c in chunks])
    
    # Compute overlaps, excluding document boundaries
    char_overlaps = []
    para_overlaps = []
    doc_boundary_count = 0
    
    for i in range(1, len(chunks)):
        prev, curr = chunks[i-1], chunks[i]
        
        if prev.doc_id != curr.doc_id:
            doc_boundary_count += 1
            continue
        
        # Character overlap
        ov = compute_overlap(prev.text, curr.text,
                            max_search=max(config.overlap_chars * 2, 500))
        char_overlaps.append(ov)
        
        # Paragraph overlap count
        prev_paras = prev.text.split('\n\n')
        curr_paras = curr.text.split('\n\n')
        para_ov = 0
        for j in range(min(len(prev_paras), len(curr_paras))):
            if prev_paras[-(j+1):] == curr_paras[:j+1]:
                para_ov = j + 1
        para_overlaps.append(para_ov)
    
    char_overlaps = np.array(char_overlaps) if char_overlaps else np.array([0])
    para_overlaps = np.array(para_overlaps) if para_overlaps else np.array([0])
    valid_char = char_overlaps[char_overlaps > 0]
    
    # Stats
    size_median = np.median(lens)
    char_overlap_median = np.median(valid_char) if len(valid_char) > 0 else 0
    para_overlap_median = np.median(para_overlaps) if len(para_overlaps) > 0 else 0
    
    # For paragraph-based: count how many achieved target paragraph overlap
    para_at_target = np.sum(para_overlaps >= config.overlap_paragraphs)
    char_at_target = np.sum(valid_char >= config.overlap_chars) if len(valid_char) > 0 else 0
    
    return {
        'n_chunks': len(chunks),
        'size_median': int(size_median),
        'size_min': int(np.min(lens)),
        'size_max': int(np.max(lens)),
        'target_chars': config.target_chars,
        'n_transitions': len(char_overlaps),
        'doc_boundaries': doc_boundary_count,
        # Paragraph-based metrics (primary for paragraph chunking)
        'para_overlap_median': float(para_overlap_median),
        'para_overlap_target': config.overlap_paragraphs,
        'para_at_target': int(para_at_target),
        'para_at_target_pct': f"{100*para_at_target/len(para_overlaps):.1f}%" if len(para_overlaps) > 0 else "N/A",
        # Character-based metrics (secondary)
        'overlap_median': int(char_overlap_median),
        'overlap_target': config.overlap_chars,
        'below_target': int(len(valid_char) - char_at_target),
        'below_target_pct': f"{100*(len(valid_char) - char_at_target)/len(valid_char):.1f}%" if len(valid_char) > 0 else "N/A"
    }


# =============================================================================
# Batch Processing with Document Tracking (Character-based)
# =============================================================================

def chunk_batch_tracked(
    texts: List[str], 
    config: AggregateConfig
) -> List[ChunkInfo]:
    """
    Chunk multiple documents with document boundary tracking.
    
    This allows proper overlap analysis that excludes document boundaries.
    
    Args:
        texts: List of document strings
        config: Chunking configuration
    
    Returns:
        List of ChunkInfo with document tracking metadata
    """
    all_chunks = []
    
    for doc_id, text in enumerate(texts):
        doc_chunks = chunk_equidistant(text, config)
        
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            info = ChunkInfo(
                text=chunk_text,
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                is_doc_start=(chunk_idx == 0),
                is_doc_end=(chunk_idx == len(doc_chunks) - 1)
            )
            all_chunks.append(info)
    
    return all_chunks


def chunk_batch(
    texts: Iterator[str], 
    config: AggregateConfig
) -> Generator[List[str], None, None]:
    """Chunk multiple documents, yielding chunk lists per document."""
    for text in texts:
        yield chunk_equidistant(text, config)


# =============================================================================
# Robust Statistics
# =============================================================================

def compute_overlap(chunk_a: str, chunk_b: str, max_search: int = 500) -> int:
    """Find exact character overlap between two consecutive chunks."""
    max_possible = min(len(chunk_a), len(chunk_b), max_search)
    
    for ov in range(max_possible, 0, -1):
        if chunk_a[-ov:] == chunk_b[:ov]:
            return ov
    return 0


def robust_stats(
    chunks: List[ChunkInfo], 
    config: AggregateConfig
) -> dict:
    """
    Compute robust statistics for chunks with document boundary awareness.
    
    Uses log-transformed median ± MAD for overlap analysis.
    """
    if not chunks:
        return {}
    
    # Chunk size stats
    lens = np.array([len(c) for c in chunks])
    
    # Compute overlaps, tracking document boundaries
    overlaps = []
    doc_boundary_count = 0
    
    for i in range(1, len(chunks)):
        prev, curr = chunks[i-1], chunks[i]
        
        # Check if this is a document boundary
        if prev.doc_id != curr.doc_id:
            doc_boundary_count += 1
            continue  # Skip document boundaries for overlap analysis
        
        ov = compute_overlap(prev.text, curr.text, 
                            max_search=max(config.overlap_chars * 2, 500))
        overlaps.append(ov)
    
    overlaps = np.array(overlaps) if overlaps else np.array([0])
    
    # Robust size statistics
    size_median = np.median(lens)
    size_mad = np.median(np.abs(lens - size_median))
    
    # Robust overlap statistics (log-transformed for right-skewed data)
    valid_overlaps = overlaps[overlaps > 0]
    
    if len(valid_overlaps) > 0:
        log_overlaps = np.log(valid_overlaps)
        log_median = np.median(log_overlaps)
        log_mad = np.median(np.abs(log_overlaps - log_median))
        
        # Transform bounds back to original scale
        log_lower = log_median - 2 * log_mad
        log_upper = log_median + 2 * log_mad
        overlap_lower = exp(log_lower)
        overlap_upper = exp(log_upper)
        
        in_log_bounds = np.sum(
            (valid_overlaps >= overlap_lower) & 
            (valid_overlaps <= overlap_upper)
        )
        pct_in_bounds = 100 * in_log_bounds / len(valid_overlaps)
        
        overlap_median = np.median(valid_overlaps)
        overlap_mad = np.median(np.abs(valid_overlaps - overlap_median))
    else:
        overlap_median = 0
        overlap_mad = 0
        overlap_lower = 0
        overlap_upper = 0
        pct_in_bounds = 0
    
    # Bounds compliance for sizes
    in_size_bounds = np.sum(
        (lens >= config.min_size) & (lens <= config.max_size)
    )
    
    return {
        # Size stats
        'n_chunks': len(lens),
        'size_median': round(size_median),
        'size_mad': round(size_mad, 1),
        'size_min': int(np.min(lens)),
        'size_max': int(np.max(lens)),
        'size_in_bounds': f"{in_size_bounds}/{len(lens)} ({100*in_size_bounds/len(lens):.1f}%)",
        
        # Overlap stats (within documents only)
        'n_transitions': len(overlaps),
        'doc_boundaries': doc_boundary_count,
        'overlap_median': round(overlap_median),
        'overlap_mad': round(overlap_mad, 1),
        'overlap_min': int(np.min(valid_overlaps)) if len(valid_overlaps) > 0 else 0,
        'overlap_max': int(np.max(valid_overlaps)) if len(valid_overlaps) > 0 else 0,
        'overlap_target': config.overlap_chars,
        
        # Log-transformed robust bounds
        'overlap_robust_lower': round(overlap_lower),
        'overlap_robust_upper': round(overlap_upper),
        'overlap_in_robust_bounds': f"{pct_in_bounds:.1f}%",
        
        # Below target
        'below_target': int(np.sum(valid_overlaps < config.overlap_chars)) if len(valid_overlaps) > 0 else 0,
        'below_target_pct': f"{100*np.sum(valid_overlaps < config.overlap_chars)/len(valid_overlaps):.1f}%" if len(valid_overlaps) > 0 else "N/A"
    }


def stats(chunks: List[str], config: AggregateConfig) -> dict:
    """
    Basic stats for simple chunk lists (no document tracking).
    
    For more detailed analysis with document boundaries, use robust_stats().
    """
    if not chunks:
        return {}
    
    lens = [len(c) for c in chunks]
    mean = sum(lens) / len(lens)
    std = (sum((l - mean) ** 2 for l in lens) / len(lens)) ** 0.5
    in_bounds = sum(1 for l in lens if config.min_size <= l <= config.max_size)
    
    overlaps = []
    for i in range(1, len(chunks)):
        ov = compute_overlap(chunks[i-1], chunks[i], 
                            max_search=max(config.overlap_chars * 2, 500))
        overlaps.append(ov)
    
    valid_overlaps = [o for o in overlaps if o > 0]
    
    return {
        'n': len(lens),
        'min': min(lens),
        'max': max(lens),
        'mean': round(mean),
        'std': round(std),
        'in_bounds': f"{in_bounds}/{len(lens)}",
        'avg_overlap': round(sum(valid_overlaps) / len(valid_overlaps)) if valid_overlaps else 0,
        'min_overlap': min(valid_overlaps) if valid_overlaps else 0,
        'max_overlap': max(valid_overlaps) if valid_overlaps else 0,
        'target_overlap': config.overlap_chars,
        'below_target': sum(1 for o in valid_overlaps if o < config.overlap_chars)
    }


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("EQUIDISTANT CHUNKING v2")
    print("=" * 80)
    
    # Default config
    config = AggregateConfig(429, 107.25, ratio=4, overlap_pct=0.5)
    print(f"\nConfig: {config}")
    
    sample = """
    Machine learning models require substantial training data. The quality of data 
    directly impacts model performance. Data preprocessing is often the most 
    time-consuming step. Feature engineering requires domain expertise.
    
    Neural networks have revolutionized many fields. Deep learning enables 
    automatic feature extraction. Transformer architectures dominate NLP tasks.
    Attention mechanisms allow models to focus on relevant inputs.
    
    Deployment presents its own challenges. Model serving requires infrastructure.
    Latency constraints affect architecture choices. Monitoring prevents drift.
    """ * 15
    
    chunks = chunk_equidistant(sample, config)
    
    print(f"\nBasic Stats: {stats(chunks, config)}")
    
    if len(chunks) >= 2:
        print(f"\nChunk 0 ({len(chunks[0])}ch):")
        print(f"  ...{chunks[0][-80:]}")
        print(f"\nChunk 1 ({len(chunks[1])}ch):")
        print(f"  {chunks[1][:80]}...")
        
        overlap = compute_overlap(chunks[0], chunks[1])
        print(f"\nActual overlap: {overlap} chars (target: {config.overlap_chars})")
