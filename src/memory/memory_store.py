import sqlite3
import json
import os
from typing import List, Optional, Dict
from datetime import datetime
from .memory_types import Memory, MemoryType

class MemoryStore:
    """SQLite-based memory storage"""
    
    def __init__(self, db_path: str = "../data/memory/ltm.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        self._enable_wal_mode()
    
    def _get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return conn
    
    def _enable_wal_mode(self):
        """Enable Write-Ahead Logging for better concurrency"""
        # Disabled WAL mode to avoid locking issues
        pass
    
    def _init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if memories table exists
        cursor.execute('''
            SELECT name FROM sqlite_master WHERE type='table' AND name='memories'
        ''')
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # Create new table with all columns
            cursor.execute('''
                CREATE TABLE memories (
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
                    related_memory_ids TEXT,
                    session_id TEXT,
                    user_id TEXT
                )
            ''')
        else:
            # Check if new columns exist and add them if they don't
            cursor.execute("PRAGMA table_info(memories)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'session_id' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN session_id TEXT')
                print("✅ Added session_id column to memories table")
            
            if 'user_id' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN user_id TEXT')
                print("✅ Added user_id column to memories table")
        
        # Create indexes for efficient querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)
        ''')
        
        # Sessions tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                memory_count INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_sessions ON sessions(user_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_memory(self, memory: Memory):
        """Store a memory"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Extract session_id and user_id from metadata if present
            session_id = memory.metadata.get('session_id') if memory.metadata else None
            user_id = memory.metadata.get('user_id') if memory.metadata else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(memory.related_memory_ids),
                session_id,
                user_id
            ))
            
            # Update session tracking within the same connection
            if session_id and user_id:
                now = datetime.now().isoformat()
                
                # Check if session exists
                cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (session_id,))
                exists = cursor.fetchone()
                
                if exists:
                    cursor.execute('''
                        UPDATE sessions 
                        SET last_active = ?, memory_count = (
                            SELECT COUNT(*) FROM memories WHERE session_id = ?
                        )
                        WHERE session_id = ?
                    ''', (now, session_id, session_id))
                else:
                    cursor.execute('''
                        INSERT INTO sessions (session_id, user_id, created_at, last_active, memory_count)
                        VALUES (?, ?, ?, ?, 1)
                    ''', (session_id, user_id, now, now))
            
            conn.commit()
        finally:
            conn.close()
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories WHERE memory_id = ?', (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_memory(row)
        return None
    
    def update_memory_access(self, memory_id: str, new_strength: Optional[float] = None):
        """Update memory access timestamp, count, and optionally strength
        
        Args:
            memory_id: ID of the memory to update
            new_strength: Optional new strength value (for decay application)
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            if new_strength is not None:
                cursor.execute('''
                    UPDATE memories 
                    SET last_accessed = ?, access_count = access_count + 1, strength = ?
                    WHERE memory_id = ?
                ''', (datetime.now().isoformat(), new_strength, memory_id))
            else:
                cursor.execute('''
                    UPDATE memories 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE memory_id = ?
                ''', (datetime.now().isoformat(), memory_id))
            
            conn.commit()
        finally:
            conn.close()
    
    def batch_update_memory_access(self, updates: List[tuple]):
        """Batch update memory access for multiple memories
        
        Args:
            updates: List of (memory_id, new_strength) tuples
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            for memory_id, new_strength in updates:
                cursor.execute('''
                    UPDATE memories 
                    SET last_accessed = ?, access_count = access_count + 1, strength = ?
                    WHERE memory_id = ?
                ''', (now, new_strength, memory_id))
            
            conn.commit()
        finally:
            conn.close()
    
    def update_memory_strength(self, memory_id: str, new_strength: float):
        """Update memory strength"""
        conn = self._get_connection()
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
        conn = self._get_connection()
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
    
    def get_memories_by_session(
        self,
        session_id: str,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0,
        limit: Optional[int] = None
    ) -> List[Memory]:
        """Get memories for a specific session"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM memories WHERE session_id = ? AND strength >= ?'
        params = [session_id, min_strength]
        
        if memory_type:
            query += ' AND memory_type = ?'
            params.append(memory_type.value)
        
        query += ' ORDER BY created_at DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def get_memories_by_user(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0
    ) -> List[Memory]:
        """Get all memories for a specific user across sessions"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM memories WHERE user_id = ? AND strength >= ?'
        params = [user_id, min_strength]
        
        if memory_type:
            query += ' AND memory_type = ?'
            params.append(memory_type.value)
        
        query += ' ORDER BY created_at DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def delete_memory(self, memory_id: str):
        """Delete a memory"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM memories WHERE memory_id = ?', (memory_id,))
        conn.commit()
        conn.close()
    
    def delete_session_memories(self, session_id: str, memory_type: Optional[MemoryType] = None) -> int:
        """Delete all memories for a session. Returns count of deleted memories."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute(
                'DELETE FROM memories WHERE session_id = ? AND memory_type = ?',
                (session_id, memory_type.value)
            )
        else:
            cursor.execute('DELETE FROM memories WHERE session_id = ?', (session_id,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count
    
    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user across all sessions. Returns count of deleted memories."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM memories WHERE user_id = ?', (user_id,))
        deleted_count = cursor.rowcount
        
        # Also delete session records
        cursor.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        return deleted_count
    
    def get_memory_count(self) -> int:
        """Get total memory count"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM memories')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*), AVG(importance), AVG(strength)
            FROM memories WHERE session_id = ?
        ''', (session_id,))
        total, avg_importance, avg_strength = cursor.fetchone()
        
        cursor.execute('''
            SELECT memory_type, COUNT(*)
            FROM memories WHERE session_id = ?
            GROUP BY memory_type
        ''', (session_id,))
        type_counts = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_memories": total or 0,
            "episodic_memories": type_counts.get("episodic", 0),
            "semantic_memories": type_counts.get("semantic", 0),
            "procedural_memories": type_counts.get("procedural", 0),
            "average_importance": avg_importance or 0,
            "average_strength": avg_strength or 0
        }
    
    def get_all_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get all sessions, optionally filtered by user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT session_id, user_id, created_at, last_active, memory_count, metadata
                FROM sessions WHERE user_id = ?
                ORDER BY last_active DESC
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT session_id, user_id, created_at, last_active, memory_count, metadata
                FROM sessions
                ORDER BY last_active DESC
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "user_id": row[1],
                "created_at": row[2],
                "last_active": row[3],
                "memory_count": row[4],
                "metadata": json.loads(row[5]) if row[5] else {}
            })
        
        return sessions
    
    def get_unique_users(self) -> List[str]:
        """Get list of unique user IDs"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT user_id FROM sessions ORDER BY user_id')
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users
    
    def _update_session(self, session_id: str, user_id: str):
        """Update or create session tracking"""
        if not session_id or not user_id:
            return  # Skip if either is missing
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Check if session exists
        cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (session_id,))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute('''
                UPDATE sessions 
                SET last_active = ?, memory_count = (
                    SELECT COUNT(*) FROM memories WHERE session_id = ?
                )
                WHERE session_id = ?
            ''', (now, session_id, session_id))
        else:
            cursor.execute('''
                INSERT INTO sessions (session_id, user_id, created_at, last_active, memory_count)
                VALUES (?, ?, ?, ?, 1)
            ''', (session_id, user_id, now, now))
        
        conn.commit()
        conn.close()
    
    def migrate_existing_data(self):
        """Migrate existing memories to extract session_id and user_id from metadata"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT memory_id, metadata FROM memories WHERE session_id IS NULL')
        rows = cursor.fetchall()
        
        updated = 0
        for memory_id, metadata_json in rows:
            if metadata_json:
                metadata = json.loads(metadata_json)
                session_id = metadata.get('session_id')
                user_id = metadata.get('user_id')
                
                if session_id or user_id:
                    cursor.execute('''
                        UPDATE memories SET session_id = ?, user_id = ? WHERE memory_id = ?
                    ''', (session_id, user_id, memory_id))
                    updated += 1
                    
                    # Create session entry
                    if session_id and user_id:
                        cursor.execute('''
                            INSERT OR IGNORE INTO sessions (session_id, user_id, created_at, last_active, memory_count)
                            VALUES (?, ?, datetime('now'), datetime('now'), 0)
                        ''', (session_id, user_id))
        
        conn.commit()
        conn.close()
        
        return updated
    
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
