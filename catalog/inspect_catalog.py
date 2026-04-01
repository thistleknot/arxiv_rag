"""Inspect feature catalog sqlite3 files."""

import sqlite3
import json

# Check all 4 sqlite3 files
files = [
    'bio_tagger_features.sqlite3',
    'graph_transformer_feature_catalog.sqlite3',
    'feature_catalog.sqlite3',
    'retriever_feature_catalog.sqlite3'
]

for db_file in files:
    print(f"\n{'='*60}")
    print(f"DATABASE: {db_file}")
    print('='*60)
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\nTables: {tables}")
        
        # For each table, show structure and sample rows
        for table in tables:
            print(f"\n--- {table} ---")
            
            # Get schema
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            print(f"Columns: {[col[1] for col in columns]}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Row count: {count}")
            
            # Show a few rows if count > 0
            if count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                rows = cursor.fetchall()
                for i, row in enumerate(rows, 1):
                    print(f"\nRow {i}:")
                    for col_info, value in zip(columns, row):
                        col_name = col_info[1]
                        # Truncate long text
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + '...'
                        print(f"  {col_name}: {value}")
        
        conn.close()
    
    except Exception as e:
        print(f"Error reading {db_file}: {e}")
