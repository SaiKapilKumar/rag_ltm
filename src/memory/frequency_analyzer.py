from typing import List, Dict, Optional
from collections import Counter
import re
import numpy as np
from .memory_store import MemoryStore
from .memory_types import MemoryType


class FrequencyAnalyzer:
    """Analyze query patterns and extract frequent topics from memories"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        
        # Common stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
            'their', 'am', 'get', 'make', 'take', 'go', 'come', 'know', 'think',
            'see', 'look', 'want', 'use', 'find', 'give', 'tell', 'work', 'call'
        }
    
    def get_frequent_topics(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_n: int = 10,
        min_word_length: int = 3
    ) -> List[Dict]:
        """Extract most frequent topics from queries
        
        Args:
            session_id: Filter by specific session
            user_id: Filter by specific user
            top_n: Number of top topics to return
            min_word_length: Minimum length for words to consider
            
        Returns:
            List of dicts with 'topic', 'count', 'percentage'
        """
        # Get relevant memories (episodic only, as they contain Q&A)
        if session_id:
            memories = self.memory_store.get_memories_by_session(
                session_id, 
                memory_type=MemoryType.EPISODIC
            )
        elif user_id:
            memories = self.memory_store.get_memories_by_user(
                user_id,
                memory_type=MemoryType.EPISODIC
            )
        else:
            memories = self.memory_store.get_all_memories(
                memory_type=MemoryType.EPISODIC
            )
        
        if not memories:
            return []
        
        # Extract queries from memory content (format: "Q: question\nA: answer")
        queries = []
        for memory in memories:
            query = self._extract_query(memory.content)
            if query:
                queries.append(query)
        
        # Extract keywords from queries
        all_keywords = []
        for query in queries:
            keywords = self._extract_keywords(query, min_word_length)
            all_keywords.extend(keywords)
        
        # Count frequencies
        keyword_counts = Counter(all_keywords)
        total_count = sum(keyword_counts.values())
        
        # Get top N
        top_keywords = keyword_counts.most_common(top_n)
        
        return [
            {
                "topic": keyword,
                "count": count,
                "percentage": (count / total_count * 100) if total_count > 0 else 0
            }
            for keyword, count in top_keywords
        ]
    
    def get_query_patterns(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict:
        """Analyze query patterns and return statistics
        
        Returns:
            Dict with pattern analysis including query types, lengths, timing
        """
        if session_id:
            memories = self.memory_store.get_memories_by_session(
                session_id,
                memory_type=MemoryType.EPISODIC
            )
        elif user_id:
            memories = self.memory_store.get_memories_by_user(
                user_id,
                memory_type=MemoryType.EPISODIC
            )
        else:
            memories = self.memory_store.get_all_memories(
                memory_type=MemoryType.EPISODIC
            )
        
        if not memories:
            return {
                "total_queries": 0,
                "avg_query_length": 0,
                "question_types": {},
                "queries_by_day": {},
                "most_active_times": []
            }
        
        queries = []
        query_lengths = []
        question_types = Counter()
        queries_by_day = Counter()
        query_hours = []
        
        for memory in memories:
            query = self._extract_query(memory.content)
            if query:
                queries.append(query)
                query_lengths.append(len(query.split()))
                
                # Classify question type
                q_type = self._classify_question_type(query)
                question_types[q_type] += 1
                
                # Track by day
                day = memory.created_at.date().isoformat()
                queries_by_day[day] += 1
                
                # Track time of day
                query_hours.append(memory.created_at.hour)
        
        # Find most active hours
        hour_counts = Counter(query_hours)
        most_active_times = [
            {"hour": hour, "count": count}
            for hour, count in hour_counts.most_common(5)
        ]
        
        return {
            "total_queries": len(queries),
            "avg_query_length": np.mean(query_lengths) if query_lengths else 0,
            "question_types": dict(question_types),
            "queries_by_day": dict(queries_by_day),
            "most_active_times": most_active_times
        }
    
    def get_conversation_timeline(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get chronological conversation timeline for a session
        
        Returns:
            List of dicts with query, answer, timestamp
        """
        memories = self.memory_store.get_memories_by_session(
            session_id,
            memory_type=MemoryType.EPISODIC,
            limit=limit
        )
        
        timeline = []
        for memory in memories:
            query, answer = self._extract_qa_pair(memory.content)
            if query:
                timeline.append({
                    "timestamp": memory.created_at.isoformat(),
                    "query": query,
                    "answer": answer,
                    "importance": memory.importance,
                    "access_count": memory.access_count
                })
        
        return timeline
    
    def _extract_query(self, content: str) -> Optional[str]:
        """Extract query from memory content (format: 'Q: query\nA: answer')"""
        match = re.match(r'Q:\s*(.+?)(?:\nA:|$)', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_qa_pair(self, content: str) -> tuple:
        """Extract both query and answer from memory content"""
        parts = content.split('\nA:', 1)
        query = parts[0].replace('Q:', '').strip() if parts else ""
        answer = parts[1].strip() if len(parts) > 1 else ""
        return query, answer
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text"""
        # Convert to lowercase and split into words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words
            if len(word) >= min_length and word not in self.stop_words
        ]
        
        return keywords
    
    def _classify_question_type(self, query: str) -> str:
        """Classify question by type (what, how, why, etc.)"""
        query_lower = query.lower()
        
        if query_lower.startswith('what'):
            return 'what'
        elif query_lower.startswith('how'):
            return 'how'
        elif query_lower.startswith('why'):
            return 'why'
        elif query_lower.startswith('when'):
            return 'when'
        elif query_lower.startswith('where'):
            return 'where'
        elif query_lower.startswith('who'):
            return 'who'
        elif query_lower.startswith('can') or query_lower.startswith('could'):
            return 'capability'
        elif query_lower.startswith('is') or query_lower.startswith('are') or query_lower.startswith('do'):
            return 'yes/no'
        else:
            return 'other'
    
    def get_user_memory_usage(self, user_id: str) -> Dict:
        """Get detailed memory usage statistics for a user"""
        memories = self.memory_store.get_memories_by_user(user_id)
        sessions = self.memory_store.get_all_sessions(user_id)
        
        if not memories:
            return {
                "total_memories": 0,
                "episodic_count": 0,
                "semantic_count": 0,
                "total_sessions": 0,
                "avg_memories_per_session": 0,
                "total_storage_bytes": 0,
                "oldest_memory": None,
                "newest_memory": None
            }
        
        episodic_count = sum(1 for m in memories if m.memory_type == MemoryType.EPISODIC)
        semantic_count = sum(1 for m in memories if m.memory_type == MemoryType.SEMANTIC)
        
        # Estimate storage (content + embedding)
        total_bytes = sum(
            len(m.content.encode('utf-8')) + len(m.embedding) * 4  # 4 bytes per float
            for m in memories
        )
        
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        return {
            "total_memories": len(memories),
            "episodic_count": episodic_count,
            "semantic_count": semantic_count,
            "total_sessions": len(sessions),
            "avg_memories_per_session": len(memories) / len(sessions) if sessions else 0,
            "total_storage_bytes": total_bytes,
            "oldest_memory": sorted_memories[0].created_at.isoformat() if sorted_memories else None,
            "newest_memory": sorted_memories[-1].created_at.isoformat() if sorted_memories else None,
            "avg_strength": np.mean([m.strength for m in memories]),
            "avg_importance": np.mean([m.importance for m in memories])
        }
