# planner.py
"""
Modified Planner: Routes queries to discrete math or calculus agents only.
"""

from crewai import Crew, Task, Process
from agents import planner_agent, discrete_math_agent, calculus_agent

class Planner:
    """Handles the routing of mathematical queries to appropriate specialists."""

    @staticmethod
    def analyze_and_route(user_query: str):
        """
        Analyze the user query and determine which agent should handle it.
        Only routes to discrete_math or calculus (no unrelated category).
        
        Returns:
            dict with 'route' (discrete_math or calculus) and 'reasoning'
        """
        task = Task(
            description=f"""
            Analyze this mathematical query and determine which specialist should handle it:
            
            Query: {user_query}
            
            You must classify this query into ONE of the following categories:
            
            1. "discrete_math" - For questions about:
               - Logic and proofs
               - Set theory
               - Combinatorics (counting, permutations, combinations)
               - Graph theory
               - Number theory
               - Probability (discrete)
               - Relations and functions
               - Sequences and recurrence relations
               - Boolean algebra
            
            2. "calculus" - For questions about:
               - Limits and continuity
               - Derivatives and differentiation
               - Integrals and integration
               - Differential equations
               - Series and sequences (convergence/divergence)
               - Multivariable calculus
               - Optimization problems
               - Related rates
            
            IMPORTANT: Every query must be classified as either discrete_math or calculus.
            If a query has elements of both, choose the one that is more dominant.
            If the query is not clearly mathematical, make your best judgment based on 
            which specialist would be better suited to handle it.
            
            Respond with ONLY a JSON object in this exact format:
            {{
                "route": "discrete_math" or "calculus",
                "reasoning": "Brief explanation of why this classification was chosen"
            }}
            
            Do not include any other text, explanations, or formatting outside the JSON.
            """,
            agent=planner_agent,
            expected_output="JSON object with route and reasoning"
        )

        crew = Crew(
            agents=[planner_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        return result

class Executor:
    """Handles query execution with RAG retrieval and LLM fallback."""

    @staticmethod
    def execute_discrete_math_query(user_query: str):
        """Execute discrete math query with RAG tool usage."""
        
        task = Task(
            description=f"""
You are answering a discrete mathematics question. Follow these steps EXACTLY:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION: {user_query}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 STEP 1: RETRIEVE FROM KNOWLEDGE BASE
You MUST call the query_discrete_math_rag tool with the question.
DO NOT skip this step.

📋 STEP 2: CHECK RAG RESULTS
After calling the tool:
- If "RAG RETRIEVAL SUCCESSFUL" → Use ONLY the retrieved context to answer
- If "RAG RETRIEVAL FAILED" → Use your Mistral LLM general knowledge

✍️ STEP 3: FORMAT YOUR RESPONSE

If RAG retrieval was successful:

**📚 KNOWLEDGE SOURCE: Knowledge Base (RAG)**

**💡 ANSWER:**
[Your complete answer based on the retrieved context. Reference specific information from the sources.]

**📖 SOURCES USED:**
[List the sources from the RAG tool output]

---

If RAG retrieval failed:

**🤖 KNOWLEDGE SOURCE: Mistral LLM General Knowledge**
ℹ️ Note: Could not find relevant information in the knowledge base for this query.

**💡 ANSWER:**
[Your answer using general knowledge with step-by-step explanation]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
✓ You MUST call query_discrete_math_rag tool first
✓ If RAG context is provided, you MUST use it (not your general knowledge)
✓ Always be explicit about whether you're using RAG or LLM knowledge
✓ Include source citations when using RAG
✓ Provide clear, step-by-step explanations
""",
            agent=discrete_math_agent,
            expected_output="Complete answer with clear knowledge source indication and citations"
        )

        crew = Crew(
            agents=[discrete_math_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            print("\n🔄 EXECUTING DISCRETE MATH QUERY")
            result = crew.kickoff()
            print("✅ QUERY EXECUTION COMPLETED\n")
            return result

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}\n")
            class ErrorResult:
                def __init__(self, error_msg):
                    self.raw = f"❌ Error executing query: {error_msg}"
            return ErrorResult(str(e))

    @staticmethod
    def execute_calculus_query(user_query: str):
        """Execute calculus query with RAG tool usage."""
        
        task = Task(
            description=f"""
You are answering a calculus question. Follow these steps EXACTLY:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION: {user_query}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 STEP 1: RETRIEVE FROM KNOWLEDGE BASE
You MUST call the query_calculus_rag tool with the question.
DO NOT skip this step.

📋 STEP 2: CHECK RAG RESULTS
After calling the tool:
- If "RAG RETRIEVAL SUCCESSFUL" → Use ONLY the retrieved context to answer
- If "RAG RETRIEVAL FAILED" → Use your Mistral LLM general knowledge

✍️ STEP 3: FORMAT YOUR RESPONSE

If RAG retrieval was successful:

**📚 KNOWLEDGE SOURCE: Knowledge Base (RAG)**

**💡 ANSWER:**
[Your complete answer based on the retrieved context]

**📖 SOURCES USED:**
[List the sources from the RAG tool output]

---

If RAG retrieval failed (currently expected for calculus):

**🤖 KNOWLEDGE SOURCE: Mistral LLM General Knowledge**
ℹ️ Note: Calculus knowledge base not yet available.

**💡 ANSWER:**
[Provide complete step-by-step solution]

**📝 SOLUTION STEPS:**
1. [Step 1 with explanation]
2. [Step 2 with explanation]
...

**🎯 FINAL ANSWER:**
[Clear final answer]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
✓ You MUST call query_calculus_rag tool first
✓ Show all mathematical steps
✓ Explain reasoning for each step
✓ Be clear about using LLM knowledge
""",
            agent=calculus_agent,
            expected_output="Complete step-by-step calculus solution"
        )

        crew = Crew(
            agents=[calculus_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            print("\n🔄 EXECUTING CALCULUS QUERY")
            result = crew.kickoff()
            print("✅ QUERY EXECUTION COMPLETED\n")
            return result
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}\n")
            class ErrorResult:
                def __init__(self, error_msg):
                    self.raw = f"❌ Error executing query: {error_msg}"
            return ErrorResult(str(e))