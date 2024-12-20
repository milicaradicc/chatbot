from typing import Dict, List, Tuple
import re
from dataclasses import dataclass

@dataclass
class RAGMetrics:
    groundedness: float
    answer_relevance: float
    context_relevance: float
    
    def get_average_score(self) -> float:
        return (self.groundedness + self.answer_relevance + self.context_relevance) / 3

class RAGEvaluator:
    def __init__(self):
        self.nlp = None  # Could integrate with spacy or other NLP library if needed
        
    def evaluate_groundedness(self, response: str, context: str) -> Tuple[float, List[str]]:
        """
        Evaluates how well the response is grounded in the context.
        Returns a score and list of statements not found in context.
        """
        # Split response and context into sentences
        response_sentences = [s.strip() for s in response.split('.') if s.strip()]
        context_sentences = [s.strip() for s in context.split('.') if s.strip()]
        
        # Convert to lowercase for comparison
        context_lower = [s.lower() for s in context_sentences]
        
        ungrounded_statements = []
        grounded_count = 0
        
        for sentence in response_sentences:
            sentence_lower = sentence.lower()
            is_grounded = False
            
            # Check if the sentence or a similar version exists in context
            for context_sent in context_lower:
                # Calculate similarity (simplified version - could use more sophisticated methods)
                common_words = set(sentence_lower.split()) & set(context_sent.split())
                total_words = len(set(sentence_lower.split()))
                
                if len(common_words) / total_words > 0.5:  # Threshold for similarity
                    is_grounded = True
                    break
                    
            if is_grounded:
                grounded_count += 1
            else:
                ungrounded_statements.append(sentence)
                
        score = (grounded_count / len(response_sentences)) * 10
        return score, ungrounded_statements
    
    def evaluate_answer_relevance(self, question: str, response: str) -> float:
        """
        Evaluates how well the response answers the question.
        Returns a score out of 10.
        """
        # Simple keyword-based relevance check
        question_keywords = set(question.lower().split())
        response_keywords = set(response.lower().split())
        
        # Remove common stop words (simplified version)
        stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        question_keywords = question_keywords - stop_words
        
        # Check if key question terms are addressed in response
        covered_keywords = question_keywords & response_keywords
        keyword_coverage = len(covered_keywords) / len(question_keywords)
        
        # Additional factors could be considered here
        return keyword_coverage * 10
    
    def evaluate_context_relevance(self, question: str, context: str) -> float:
        """
        Evaluates how well the context matches the question.
        Returns a score out of 10.
        """
        question_keywords = set(question.lower().split())
        context_keywords = set(context.lower().split())
        
        # Remove stop words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        question_keywords = question_keywords - stop_words
        
        # Check keyword coverage
        covered_keywords = question_keywords & context_keywords
        keyword_coverage = len(covered_keywords) / len(question_keywords)
        
        # Consider context length (penalize if too long or too short)
        ideal_length = len(question_keywords) * 20  # Arbitrary multiplier
        length_ratio = min(len(context.split()) / ideal_length, 1.5)
        length_score = 1 - abs(1 - length_ratio) * 0.5
        
        return min(keyword_coverage * length_score * 10, 10)
    
    def evaluate(self, question: str, response: str, context: str) -> RAGMetrics:
        """
        Evaluates all three RAG metrics for the given QA pair and context.
        """
        groundedness_score, ungrounded = self.evaluate_groundedness(response, context)
        answer_relevance_score = self.evaluate_answer_relevance(question, response)
        context_relevance_score = self.evaluate_context_relevance(question, context)
        
        return RAGMetrics(
            groundedness=groundedness_score,
            answer_relevance=answer_relevance_score,
            context_relevance=context_relevance_score
        )

def main():
    # Example usage
    evaluator = RAGEvaluator()
    
    # Test case
    question = "what is python"
    response = "Python is a high-level, general-purpose programming language that supports multiple programming paradigms. It is designed to be easily readable and aims to provide a fun and enjoyable experience for developers. Additionally, Python's design and philosophy have influenced many other programming languages."
    context = "Python is a high-level, general-purpose programming language. Python is a multi-paradigm programming language. Python's developers aim for it to be fun to use. Python is meant to be an easily readable language. Python's design and philosophy have influenced many other programming languages."
    
    metrics = evaluator.evaluate(question, response, context)
    
    print(f"RAG Evaluation Results:")
    print(f"Groundedness: {metrics.groundedness:.1f}/10")
    print(f"Answer Relevance: {metrics.answer_relevance:.1f}/10")
    print(f"Context Relevance: {metrics.context_relevance:.1f}/10")
    print(f"Average Score: {metrics.get_average_score():.1f}/10")

if __name__ == "__main__":
    main()