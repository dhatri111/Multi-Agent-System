# reflective.pyy
"""
Reflector: Performs quality assessment and introspection on answers.
"""

from crewai import Crew, Task, Process
from agents import reflective_agent

class Reflector:
    """Handles quality assessment and reflection on mathematical answers."""
    
    # method to evaluate answer quality
    @staticmethod
    def evaluate_answer(user_query: str, answer: str, route: str):
        """
        Evaluate the quality of a mathematical answer. Do self-critique and introspection.
        
        Metrics from 0-100:
        - Correctness : Mathematical accuracy
        - Clarity : Clear explanation
        - Completeness : Comprehensive coverage
        - Presentation : Proper notation and formatting
        - Confidence : how confident the model is in its answer
        
        Returns quality score (0-100) and feedback.
        """
        
        task = Task(
            description=f"""
            Evaluate the quality of this mathematical answer:
            
            ORIGINAL QUESTION:
            {user_query}
            
            ANSWER PROVIDED ({route}):
            {answer}
            
            Assess the answer based on these criteria: from 0 to 100%
            1. Correctness : Is the mathematics accurate and valid?
            2. Clarity : Is the explanation clear and easy to follow?
            3. Completeness : Does it fully address the question?
            4. Presentation : Is the notation proper and formatting good?
            
            Provide your evaluation in this format:
            
            QUALITY SCORE: [0-100]
            
            DETAILED BREAKDOWN:
            - Correctness : [score] - [brief comment]
            - Clarity : [score] - [brief comment]
            - Completeness : [score] - [brief comment]
            - Presentation : [score] - [brief comment]
            Confidence : [score]- [brief comment]
            
            STRENGTHS:
            - [List key strengths]
            
            AREAS FOR IMPROVEMENT:
            - [List specific suggestions]
            
            OVERALL ASSESSMENT:
            [Brief summary of the answer quality]
            """,
            agent=reflective_agent,
            expected_output="Quality score and detailed evaluation"
        )

        crew = Crew(
            agents=[reflective_agent], # Single agent for reflection
            tasks=[task], # Single task
            process=Process.sequential, # Sequential processing
            verbose=False  # No need for verbose in reflection
        )
        # Kickoff the crew to execute the task
        return crew.kickoff()