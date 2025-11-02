class ImportanceCalculator:
    """Calculate importance scores for memories"""
    
    @staticmethod
    def calculate_importance(
        content: str,
        metadata: dict = None,
        context: str = ""
    ) -> float:
        """Calculate importance score for a memory"""
        score = 0.5
        
        # Length factor (longer content may be more detailed)
        if len(content) > 200:
            score += 0.1
        elif len(content) < 50:
            score -= 0.1
        
        # Keyword matching for important concepts
        important_keywords = [
            'important', 'critical', 'key', 'essential', 'remember',
            'prefer', 'always', 'never', 'must', 'should', 'fact'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(1 for kw in important_keywords if kw in content_lower)
        score += min(keyword_count * 0.05, 0.2)
        
        # Question marks might indicate clarification needed (lower importance)
        if content.count('?') > 2:
            score -= 0.1
        
        # Exclamation marks might indicate emphasis
        if '!' in content:
            score += 0.05
        
        # Metadata boosters
        if metadata:
            if metadata.get('user_marked_important'):
                score += 0.3
            if metadata.get('source') == 'user_fact':
                score += 0.2
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, score))
