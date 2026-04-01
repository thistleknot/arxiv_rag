"""
ArXiv Chunking Pipeline

Combines section-level aggregation from chunk_n_ingest_arxiv with
equidistant paragraph chunking for RAG-optimized text chunks.

PIPELINE:
    1. Load markdown files from papers/post_processed
    2. Filter: Remove base64/image lines, References, Appendix sections
    3. Section Aggregation: Split by ## headers, merge small sections
    4. Paragraph Statistics: Compute corpus-wide paragraph stats
    5. Equidistant Chunking: Apply within each section using corpus stats

USAGE:
    from arxiv_chunking_pipeline import ArxivChunkingPipeline
    
    pipeline = ArxivChunkingPipeline("C:/Users/user/arxiv_id_lists/papers/post_processed")
    chunks = pipeline.process()
    
    # Access results
    for chunk in chunks:
        print(f"Doc: {chunk.doc_id}, Section: {chunk.section_idx}, Chunk: {chunk.chunk_idx}")
        print(chunk.text[:200])
"""

import os
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator
from tqdm import tqdm
from scipy.stats import median_abs_deviation
from math import exp

# Import from equidistant_chunking
from equidistant_chunking import (
    AggregateConfig,
    ParagraphConfig,
    SampleStats,
    ChunkInfo,
    estimate_from_sample,
    create_paragraph_blocks,
    SlidingAggregator,
    paragraph_config_from_sample,
)

# Optional LangChain import for vector store integration
try:
    from langchain_core.documents import Document  # type: ignore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = None  # type: ignore


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class ArxivChunk:
    """Chunk with arxiv-specific metadata."""
    text: str
    doc_id: str              # Paper ID (e.g., "1301_3781")
    section_idx: int         # Section index within paper
    section_title: str       # Section header text
    chunk_idx: int           # Chunk index within section
    is_section_start: bool
    is_section_end: bool
    char_count: int = field(init=False)
    
    def __post_init__(self):
        self.char_count = len(self.text)
    
    def __len__(self):
        return self.char_count


@dataclass  
class ProcessedPaper:
    """Paper after filtering and section splitting."""
    doc_id: str
    sections: List[Tuple[str, str]]  # List of (section_title, section_content)
    original_char_count: int
    filtered_char_count: int
    sections_removed: List[str]  # Names of removed sections


# =============================================================================
# Content Filtering
# =============================================================================

class ContentFilter:
    """
    Filters unwanted content from arxiv markdown files.
    
    Removes:
    - Base64 encoded data (images)
    - Markdown image references
    - References section and everything after
    - Appendix sections
    """
    
    # Patterns for content to remove
    BASE64_PATTERN = re.compile(
        r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
        re.IGNORECASE
    )
    
    # Long base64-like strings (64+ chars of base64 alphabet)
    BASE64_STRING_PATTERN = re.compile(
        r'[A-Za-z0-9+/]{64,}={0,2}'
    )
    
    # Markdown image syntax
    IMAGE_PATTERN = re.compile(
        r'!\[([^\]]*)\]\([^)]+\)',
        re.MULTILINE
    )
    
    # Section headers that indicate end of main content
    EXCLUDED_SECTION_PATTERNS = [
        re.compile(r'^#+\s*references?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^#+\s*bibliography\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^#+\s*appendix', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^#+\s*acknowledgements?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^#+\s*supplementary', re.IGNORECASE | re.MULTILINE),
    ]
    
    @classmethod
    def filter_line(cls, line: str) -> Optional[str]:
        """
        Filter a single line, returning None if line should be removed.
        
        Args:
            line: Single line of text
            
        Returns:
            Filtered line or None if line should be removed
        """
        # Remove lines with base64 data URIs
        if cls.BASE64_PATTERN.search(line):
            return None
        
        # Remove lines that are mostly base64 strings (>50% of line)
        base64_matches = cls.BASE64_STRING_PATTERN.findall(line)
        if base64_matches:
            total_base64_len = sum(len(m) for m in base64_matches)
            if total_base64_len > len(line) * 0.5:
                return None
        
        # Remove markdown image lines (but keep caption if meaningful)
        img_match = cls.IMAGE_PATTERN.search(line)
        if img_match:
            # Check if the line is primarily just the image
            line_without_img = cls.IMAGE_PATTERN.sub('', line).strip()
            if len(line_without_img) < 20:
                return None
            return line_without_img
        
        return line
    
    @classmethod
    def filter_text(cls, text: str) -> str:
        """
        Filter all lines in text, removing unwanted content.
        
        Args:
            text: Full text content
            
        Returns:
            Filtered text
        """
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            filtered = cls.filter_line(line)
            if filtered is not None:
                filtered_lines.append(filtered)
        
        return '\n'.join(filtered_lines)
    
    @classmethod
    def find_excluded_section_start(cls, text: str) -> Tuple[int, str]:
        """
        Find the position where excluded sections (References, Appendix) begin.
        
        Args:
            text: Full document text
            
        Returns:
            Tuple of (position, section_name) or (-1, "") if not found
        """
        earliest_pos = len(text)
        earliest_section = ""
        
        for pattern in cls.EXCLUDED_SECTION_PATTERNS:
            match = pattern.search(text)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
                earliest_section = match.group().strip()
        
        if earliest_pos < len(text):
            return earliest_pos, earliest_section
        return -1, ""
    
    @classmethod
    def remove_excluded_sections(cls, text: str) -> Tuple[str, List[str]]:
        """
        Remove References, Appendix, and other excluded sections.
        
        Args:
            text: Full document text
            
        Returns:
            Tuple of (filtered_text, list_of_removed_section_names)
        """
        removed_sections = []
        
        pos, section_name = cls.find_excluded_section_start(text)
        if pos > 0:
            removed_sections.append(section_name)
            text = text[:pos].rstrip()
        
        return text, removed_sections


# =============================================================================
# Section Processing
# =============================================================================

class SectionProcessor:
    """
    Processes papers into sections and handles section-level operations.
    """
    
    SECTION_PATTERN = re.compile(r'^##\s+', re.MULTILINE)
    
    @classmethod
    def split_into_sections(cls, text: str) -> List[Tuple[str, str]]:
        """
        Split paper into sections based on ## headers.
        
        Args:
            text: Full paper text
            
        Returns:
            List of (section_title, section_content) tuples
        """
        # Split by ## headers
        parts = re.split(r'^(##\s+.+)$', text, flags=re.MULTILINE)
        
        sections = []
        current_title = "Introduction"  # Default for content before first ##
        current_content = []
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            if part.startswith('## '):
                # Save previous section if it has content
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_title, content))
                
                # Start new section
                current_title = part[3:].strip()  # Remove "## "
                current_content = []
            elif part:
                current_content.append(part)
            
            i += 1
        
        # Don't forget the last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((current_title, content))
        
        return sections
    
    @classmethod
    def merge_small_sections(
        cls,
        sections: List[Tuple[str, str]],
        min_chars: int,
        max_chars: int
    ) -> List[Tuple[str, str]]:
        """
        Merge adjacent small sections until they reach min_chars.
        
        Args:
            sections: List of (title, content) tuples
            min_chars: Minimum section size
            max_chars: Maximum section size (don't merge beyond this)
            
        Returns:
            Merged sections
        """
        if not sections:
            return sections
        
        merged = []
        i = 0
        
        while i < len(sections):
            title, content = sections[i]
            current_len = len(content)
            
            # If section is already large enough, keep as-is
            if current_len >= min_chars:
                merged.append((title, content))
                i += 1
                continue
            
            # Try to merge with subsequent small sections
            merged_content = content
            merged_titles = [title]
            j = i + 1
            
            while j < len(sections) and len(merged_content) < min_chars:
                next_title, next_content = sections[j]
                combined_len = len(merged_content) + len(next_content) + 2  # +2 for \n\n
                
                # Don't exceed max
                if combined_len > max_chars:
                    break
                
                merged_content += "\n\n" + next_content
                merged_titles.append(next_title)
                j += 1
            
            # Create merged section with combined title
            if len(merged_titles) > 1:
                combined_title = " + ".join(merged_titles)
            else:
                combined_title = title
            
            merged.append((combined_title, merged_content))
            i = j
        
        return merged


# =============================================================================
# Main Pipeline
# =============================================================================

class ArxivChunkingPipeline:
    """
    Complete pipeline for chunking arxiv papers.
    
    Combines:
    - Content filtering (base64, images, references, appendix)
    - Section-level aggregation
    - Equidistant paragraph chunking within sections
    """
    
    def __init__(
        self,
        papers_dir: str,
        target_paragraphs: int = 3,
        overlap_paragraphs: int = 1,
        min_section_chars: int = 500,
        max_section_chars: int = 10000,
        verbose: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            papers_dir: Directory containing markdown files
            target_paragraphs: Target paragraphs per chunk
            overlap_paragraphs: Paragraphs to overlap between chunks
            min_section_chars: Minimum section size before merging
            max_section_chars: Maximum section size after merging
            verbose: Print progress information
        """
        self.papers_dir = papers_dir
        self.target_paragraphs = target_paragraphs
        self.overlap_paragraphs = overlap_paragraphs
        self.min_section_chars = min_section_chars
        self.max_section_chars = max_section_chars
        self.verbose = verbose
        
        # Will be computed from corpus
        self.paragraph_config: Optional[ParagraphConfig] = None
        self.aggregate_config: Optional[AggregateConfig] = None
        self.sample_stats: Optional[SampleStats] = None
    
    def _log(self, msg: str):
        """Print if verbose mode is on."""
        if self.verbose:
            print(msg)
    
    def load_papers(self) -> List[Tuple[str, str]]:
        """
        Load all markdown files from papers directory.
        
        Returns:
            List of (doc_id, content) tuples
        """
        papers = []
        
        md_files = [f for f in os.listdir(self.papers_dir) if f.endswith('.md')]
        self._log(f"Found {len(md_files)} markdown files")
        
        iterator = tqdm(md_files, desc="Loading papers") if self.verbose else md_files
        
        for filename in iterator:
            filepath = os.path.join(self.papers_dir, filename)
            doc_id = os.path.splitext(filename)[0]
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                # Sanitize NUL bytes
                content = content.replace('\0', '')
                papers.append((doc_id, content))
            except Exception as e:
                self._log(f"Error reading {filename}: {e}")
        
        return papers
    
    def filter_papers(
        self, 
        papers: List[Tuple[str, str]]
    ) -> List[ProcessedPaper]:
        """
        Apply content filtering to all papers.
        
        Args:
            papers: List of (doc_id, content) tuples
            
        Returns:
            List of ProcessedPaper objects
        """
        processed = []
        
        iterator = tqdm(papers, desc="Filtering papers") if self.verbose else papers
        
        for doc_id, content in iterator:
            original_len = len(content)
            
            # Step 1: Remove excluded sections (References, Appendix)
            content, removed_sections = ContentFilter.remove_excluded_sections(content)
            
            # Step 2: Filter lines (base64, images)
            content = ContentFilter.filter_text(content)
            
            # Step 3: Split into sections
            sections = SectionProcessor.split_into_sections(content)
            
            # Step 4: Merge small sections
            sections = SectionProcessor.merge_small_sections(
                sections,
                self.min_section_chars,
                self.max_section_chars
            )
            
            processed.append(ProcessedPaper(
                doc_id=doc_id,
                sections=sections,
                original_char_count=original_len,
                filtered_char_count=sum(len(s[1]) for s in sections),
                sections_removed=removed_sections
            ))
        
        return processed
    
    def compute_corpus_stats(
        self, 
        processed_papers: List[ProcessedPaper]
    ) -> Tuple[ParagraphConfig, AggregateConfig, SampleStats]:
        """
        Compute paragraph statistics across the entire corpus.
        
        Stats are computed over all section content from all papers,
        ensuring consistent chunking parameters across the corpus.
        
        Args:
            processed_papers: List of ProcessedPaper objects
            
        Returns:
            Tuple of (ParagraphConfig, AggregateConfig, SampleStats)
        """
        # Collect all section content as separate "documents" for stats
        all_sections = []
        for paper in processed_papers:
            for title, content in paper.sections:
                if content.strip():
                    all_sections.append(content)
        
        self._log(f"Computing stats from {len(all_sections)} sections")
        
        # Get paragraph config from sample
        para_config, stats = paragraph_config_from_sample(
            all_sections,
            target_paragraphs=self.target_paragraphs,
            overlap_paragraphs=self.overlap_paragraphs,
            sample_size=min(500, len(all_sections))
        )
        
        # Create AggregateConfig for hybrid chunking
        agg_config = AggregateConfig(
            target=para_config.target_chars,
            tolerance=stats.para_char_mad * 2,
            ratio=max(2.0, stats.sents_per_para_median),
            overlap_pct=para_config.overlap_chars / para_config.target_chars if para_config.target_chars > 0 else 0.5,
            chars_per_word=stats.chars_per_word
        )
        
        self._log(f"Paragraph config: {para_config}")
        self._log(f"Aggregate config: {agg_config}")
        self._log(f"Sample stats: {stats.n_paragraphs} paragraphs, "
                  f"median={stats.para_char_median:.0f} chars")
        
        return para_config, agg_config, stats
    
    def chunk_section(
        self,
        content: str,
        config: AggregateConfig,
        max_para_chars: int
    ) -> List[str]:
        """
        Chunk a single section using equidistant paragraph chunking.
        
        Args:
            content: Section content
            config: AggregateConfig
            max_para_chars: Max paragraph size before splitting
            
        Returns:
            List of chunk texts
        """
        blocks = create_paragraph_blocks(content, max_para_chars)
        agg = SlidingAggregator(config)
        chunks = []
        
        for block in blocks:
            for chunk in agg.feed(block):
                chunks.append(chunk)
        
        for chunk in agg.finish(chunks):
            chunks.append(chunk)
        
        return chunks
    
    def process(self) -> List[ArxivChunk]:
        """
        Run the complete chunking pipeline.
        
        Returns:
            List of ArxivChunk objects with full metadata
        """
        # Step 1: Load papers
        papers = self.load_papers()
        if not papers:
            self._log("No papers found!")
            return []
        
        # Step 2: Filter and process
        processed_papers = self.filter_papers(papers)
        
        # Report filtering stats
        total_original = sum(p.original_char_count for p in processed_papers)
        total_filtered = sum(p.filtered_char_count for p in processed_papers)
        reduction_pct = (1 - total_filtered / total_original) * 100 if total_original > 0 else 0
        self._log(f"Filtered: {total_original:,} -> {total_filtered:,} chars "
                  f"({reduction_pct:.1f}% reduction)")
        
        # Step 3: Compute corpus-wide stats
        self.paragraph_config, self.aggregate_config, self.sample_stats = \
            self.compute_corpus_stats(processed_papers)
        
        # Step 4: Chunk each section
        all_chunks = []
        
        iterator = tqdm(processed_papers, desc="Chunking papers") if self.verbose else processed_papers
        
        for paper in iterator:
            for section_idx, (section_title, section_content) in enumerate(paper.sections):
                if not section_content.strip():
                    continue
                
                # Chunk this section
                chunk_texts = self.chunk_section(
                    section_content,
                    self.aggregate_config,
                    self.paragraph_config.max_para_chars
                )
                
                # Create ArxivChunk objects
                for chunk_idx, chunk_text in enumerate(chunk_texts):
                    chunk = ArxivChunk(
                        text=chunk_text,
                        doc_id=paper.doc_id,
                        section_idx=section_idx,
                        section_title=section_title,
                        chunk_idx=chunk_idx,
                        is_section_start=(chunk_idx == 0),
                        is_section_end=(chunk_idx == len(chunk_texts) - 1)
                    )
                    all_chunks.append(chunk)
        
        self._log(f"Created {len(all_chunks)} chunks from "
                  f"{len(processed_papers)} papers")
        
        return all_chunks
    
    def get_stats(self, chunks: List[ArxivChunk]) -> dict:
        """
        Compute statistics for the chunked output.
        
        Args:
            chunks: List of ArxivChunk objects
            
        Returns:
            Dictionary of statistics
        """
        if not chunks:
            return {}
        
        lens = np.array([c.char_count for c in chunks])
        
        # Documents and sections
        unique_docs = len(set(c.doc_id for c in chunks))
        unique_sections = len(set((c.doc_id, c.section_idx) for c in chunks))
        
        return {
            'n_chunks': len(chunks),
            'n_documents': unique_docs,
            'n_sections': unique_sections,
            'chunks_per_doc': len(chunks) / unique_docs if unique_docs > 0 else 0,
            'chunks_per_section': len(chunks) / unique_sections if unique_sections > 0 else 0,
            'size_median': int(np.median(lens)),
            'size_mean': int(np.mean(lens)),
            'size_std': int(np.std(lens)),
            'size_min': int(np.min(lens)),
            'size_max': int(np.max(lens)),
            'target_chars': self.aggregate_config.target if self.aggregate_config else None,
            'overlap_chars': self.aggregate_config.overlap_chars if self.aggregate_config else None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def chunk_arxiv_papers(
    papers_dir: str = r"C:\Users\user\arxiv_id_lists\papers\post_processed",
    target_paragraphs: int = 3,
    overlap_paragraphs: int = 1,
    verbose: bool = True
) -> List[ArxivChunk]:
    """
    Convenience function to chunk arxiv papers.
    
    Args:
        papers_dir: Directory containing markdown files
        target_paragraphs: Target paragraphs per chunk
        overlap_paragraphs: Paragraphs to overlap
        verbose: Print progress
        
    Returns:
        List of ArxivChunk objects
    """
    pipeline = ArxivChunkingPipeline(
        papers_dir=papers_dir,
        target_paragraphs=target_paragraphs,
        overlap_paragraphs=overlap_paragraphs,
        verbose=verbose
    )
    return pipeline.process()


def chunks_to_dataframe(chunks: List[ArxivChunk]):
    """
    Convert chunks to a pandas DataFrame for analysis.
    
    Args:
        chunks: List of ArxivChunk objects
        
    Returns:
        pandas DataFrame with chunk data
    """
    import pandas as pd
    
    data = []
    for chunk in chunks:
        data.append({
            'doc_id': chunk.doc_id,
            'section_idx': chunk.section_idx,
            'section_title': chunk.section_title,
            'chunk_idx': chunk.chunk_idx,
            'char_count': chunk.char_count,
            'is_section_start': chunk.is_section_start,
            'is_section_end': chunk.is_section_end,
            'text': chunk.text,
        })
    
    return pd.DataFrame(data)


def save_chunks_to_csv(
    chunks: List[ArxivChunk],
    output_path: str = "arxiv_chunks.csv"
):
    """
    Save chunks to CSV file.
    
    Args:
        chunks: List of ArxivChunk objects
        output_path: Path to output CSV file
    """
    df = chunks_to_dataframe(chunks)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(chunks)} chunks to {output_path}")


def chunks_to_langchain_documents(chunks: List[ArxivChunk]):
    """
    Convert ArxivChunks to LangChain Document objects.
    
    Args:
        chunks: List of ArxivChunk objects
        
    Returns:
        List of LangChain Document objects
        
    Raises:
        ImportError: If langchain_core is not installed
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain_core is required for this function. "
            "Install with: pip install langchain-core"
        )
    
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.text,
            metadata={
                "doc_id": chunk.doc_id,
                "section_idx": chunk.section_idx,
                "section_title": chunk.section_title,
                "chunk_idx": chunk.chunk_idx,
                "char_count": chunk.char_count,
                "is_section_start": chunk.is_section_start,
                "is_section_end": chunk.is_section_end,
                "source_type": "arxiv_paper"
            }
        )
        documents.append(doc)
    
    return documents


def create_chunk_id(chunk: ArxivChunk) -> str:
    """
    Create a unique ID for a chunk.
    
    Format: {doc_id}_{section_idx}_{chunk_idx}
    
    Args:
        chunk: ArxivChunk object
        
    Returns:
        Unique ID string
    """
    return f"{chunk.doc_id}_{chunk.section_idx}_{chunk.chunk_idx}"


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("ARXIV CHUNKING PIPELINE")
    print("=" * 80)
    
    # Run pipeline
    pipeline = ArxivChunkingPipeline(
        papers_dir=r"C:\Users\user\arxiv_id_lists\papers\post_processed",
        target_paragraphs=3,
        overlap_paragraphs=1,
        min_section_chars=500,
        max_section_chars=10000,
        verbose=True
    )
    
    chunks = pipeline.process()
    
    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    stats = pipeline.get_stats(chunks)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample chunks
    if chunks:
        print("\n" + "=" * 80)
        print("SAMPLE CHUNKS")
        print("=" * 80)
        
        # Show first 3 chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i} ---")
            print(f"Doc: {chunk.doc_id}")
            print(f"Section: {chunk.section_idx} - {chunk.section_title}")
            print(f"Chunk: {chunk.chunk_idx} ({chunk.char_count} chars)")
            print(f"Text preview: {chunk.text[:200]}...")
