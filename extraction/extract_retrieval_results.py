"""
Extract full retrieval results to markdown
"""
import msgpack

# Load chunks
with open('checkpoints/chunks.msgpack', 'rb') as f:
    chunks = msgpack.load(f, strict_map_key=False)

# Retrieved chunk IDs from query
results = [23303, 29219, 23304, 29218, 22207, 29217, 133730, 29216, 121613, 113430, 20930, 96595, 34262]

# Write to markdown
with open('agentic_memory_retrieval_results.md', 'w', encoding='utf-8') as out:
    out.write('# Three-Layer φ-Retrieval Results\n\n')
    out.write('**Query:** "agentic memory methods"\n\n')
    out.write('**Date:** 2026-02-08\n\n')
    out.write(f'**Retrieved:** {len(results)} chunks\n\n')
    out.write('---\n\n')
    
    for i, chunk_id in enumerate(results):
        chunk = chunks[chunk_id]
        out.write(f'## Chunk {i+1}: ID {chunk_id}\n\n')
        out.write(f'**Paper ID:** {chunk["paper_id"]}\n\n')
        out.write(f'**Section Index:** {chunk.get("section_idx", "N/A")}\n\n')
        
        # Extract metadata if available
        if 'arxiv_id' in chunk:
            out.write(f'**ArXiv ID:** {chunk["arxiv_id"]}\n\n')
        
        out.write('**Full Text:**\n\n')
        out.write(f'{chunk["text"]}\n\n')
        out.write('---\n\n')

print('✓ Saved to agentic_memory_retrieval_results.md')
