import sqlite3
import json
import os
from typing import List, Optional
from datetime import datetime
from .memory_types import Memory, MemoryType

class MemoryStore:
    """SQLite-based memory storage"""
    
    def __init__(self, db_path: str = "../data/memory/ltm.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER NOT NULL,
                importance REAL NOT NULL,
                strength REAL NOT NULL,
                source TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                related_memory_ids TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_memory(self, memory: Memory):
        """Store a memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.memory_id,
            memory.content,
            memory.memory_type.value,
            json.dumps(memory.embedding),
            memory.created_at.isoformat(),
            memory.last_accessed.isoformat(),
            memory.access_count,
            memory.importance,
            memory.strength,
            memory.source,
            json.dumps(memory.tags),
            json.dumps(memory.metadata),
            json.dumps(memory.related_memory_ids)
        ))
        
        conn.commit()
        conn.close()
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories WHERE memory_id = ?', (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_memory(row)
        return None
    
    def update_memory_access(self, memory_id: str):
        """Update memory access timestamp and count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE memories 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE memory_id = ?
        ''', (datetime.now().isoformat(), memory_id))
        
        conn.commit()
        conn.close()
    
    def update_memory_strength(self, memory_id: str, new_strength: float):
        """Update memory strength"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE memories SET strength = ? WHERE memory_id = ?
        ''', (new_strength, memory_id))
        
        conn.commit()
        conn.close()
    
    def get_all_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0
    ) -> List[Memory]:
        """Get all memories with optional filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM memories WHERE strength >= ?'
        params = [min_strength]
        
        if memory_type:
            query += ' AND memory_type = ?'
            params.append(memory_type.value)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def delete_memory(self, memory_id: str):
        """Delete a memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM memories WHERE memory_id = ?', (memory_id,))
        conn.commit()
        conn.close()
    
    def get_memory_count(self) -> int:
        """Get total memory count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM memories')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            memory_id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]),
            embedding=json.loads(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            last_accessed=datetime.fromisoformat(row[5]),
            access_count=row[6],
            importance=row[7],
            strength=row[8],
            source=row[9],
            tags=json.loads(row[10]) if row[10] else [],
            metadata=json.loads(row[11]) if row[11] else {},
            related_memory_ids=json.loads(row[12]) if row[12] else []
        )
