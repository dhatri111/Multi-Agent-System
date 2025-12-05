# app.py
"""
Modified Streamlit Application for Multi-Agent Math Inquiry System
with Periodic Reflection and No Unrelated Query Handling
"""

import streamlit as st
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from planner import Planner, Executor
from reflective import Reflector

# Initialize session state for reflection scheduling
if 'last_reflection_date' not in st.session_state:
    st.session_state.last_reflection_date = None
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'reflection_interval_days' not in st.session_state:
    st.session_state.reflection_interval_days = 7  # Default: weekly reflection

# Page configuration
st.set_page_config(
    page_title="Math Inquiry AI System",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .route-badge {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .discrete-math {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .calculus {
        background-color: #f3e5f5;
        color: #6a1b9a;
    }
    .reflection-info {
        background-color: #fff3cd;
        colour: black;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🧮 Multi-Agent Math Inquiry System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by RAG & Mistral LLM with Periodic Quality Assessment</p>', unsafe_allow_html=True)

# Sidebar for reflection settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Reflection Schedule")
    reflection_interval = st.selectbox(
        "Reflection Frequency:",
        options=[7, 14, 30],
        format_func=lambda x: f"Every {x} days" if x != 7 else "Weekly (7 days)",
        index=0
    )
    st.session_state.reflection_interval_days = reflection_interval
    
    # Display reflection status
    st.markdown("---")
    st.subheader("📊 Reflection Status")
    
    if st.session_state.last_reflection_date:
        last_date = st.session_state.last_reflection_date
        next_date = last_date + timedelta(days=st.session_state.reflection_interval_days)
        days_until = (next_date - datetime.now()).days
        
        st.info(f"**Last Reflection:** {last_date.strftime('%Y-%m-%d')}")
        st.info(f"**Next Reflection:** {next_date.strftime('%Y-%m-%d')}")
        st.info(f"**Days Until Next:** {days_until}")
    else:
        st.warning("No reflection performed yet")
    
    st.metric("Queries Processed", st.session_state.query_count)
    
    # Manual reflection trigger
    st.markdown("---")
    if st.button("🔍 Run Reflection Now", type="secondary"):
        st.session_state.force_reflection = True
        st.rerun()

# Function to check if reflection should run
def should_run_reflection():
    """Determine if reflection should run based on schedule."""
    # Check for manual trigger
    if st.session_state.get('force_reflection', False):
        st.session_state.force_reflection = False
        return True
    
    # Check if never run before
    if st.session_state.last_reflection_date is None:
        return True
    
    # Check if interval has passed
    days_since_last = (datetime.now() - st.session_state.last_reflection_date).days
    return days_since_last >= st.session_state.reflection_interval_days

# Main interface
st.markdown("### 💭 Ask Your Mathematical Question:")

# Query input
user_query = st.text_area(
    "Enter your question:",
    value=st.session_state.get('query', ''),
    height=100,
    placeholder="E.g., 'What is the derivative of x^2?' or 'How many ways can I arrange 5 books?'"
)

# Process button
if st.button("🚀 Submit Query", type="primary"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a question first!")
    else:
        # Increment query counter
        st.session_state.query_count += 1
        
        # Step 1: Planning
        st.markdown("---")
        st.markdown("### 📋 Step 1: Query Analysis & Routing")
        
        with st.spinner("Analyzing query..."):
            try:
                planning_result = Planner.analyze_and_route(user_query)
            except Exception as e:
                st.error(f"❌ Error during query analysis: {str(e)}")
                st.stop()
        
        # Parse the planning result
        try:
            # Extract text from CrewOutput
            if hasattr(planning_result, 'raw'):
                result_text = planning_result.raw
            elif hasattr(planning_result, 'tasks_output') and planning_result.tasks_output:
                result_text = planning_result.tasks_output[0].raw
            else:
                result_text = str(planning_result)
            
            # Try to find JSON in the text
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                route_info = json.loads(json_match.group())
            else:
                # Fallback: try to determine route from text
                result_lower = result_text.lower()
                if 'discrete' in result_lower:
                    route_info = {"route": "discrete_math", "reasoning": result_text}
                else:
                    route_info = {"route": "calculus", "reasoning": result_text}
        except Exception as e:
            st.error(f"❌ Error parsing routing decision: {str(e)}")
            # Default to calculus as fallback
            route_info = {"route": "calculus", "reasoning": "Could not parse routing decision - defaulting to calculus"}
        
        route = route_info.get("route", "calculus")
        reasoning = route_info.get("reasoning", "No reasoning provided")
        
        # Display routing decision
        col1, col2 = st.columns([1, 3])
        with col1:
            if route == "discrete_math":
                st.markdown('<span class="route-badge discrete-math">📊 Discrete Math</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="route-badge calculus">📈 Calculus</span>', unsafe_allow_html=True)
        
        with col2:
            st.info(f"**Reasoning:** {reasoning}")
        
        # Step 2: Execution
        st.markdown("### ⚙️ Step 2: Query Execution")
        
        with st.spinner(f"Processing with {route.replace('_', ' ').title()} Agent..."):
            if route == "discrete_math":
                execution_result = Executor.execute_discrete_math_query(user_query)
            else:  # calculus
                execution_result = Executor.execute_calculus_query(user_query)
        
        # Extract answer text from CrewOutput
        if hasattr(execution_result, 'raw'):
            answer_text = execution_result.raw
        elif hasattr(execution_result, 'tasks_output') and execution_result.tasks_output:
            answer_text = execution_result.tasks_output[0].raw
        else:
            answer_text = str(execution_result)
        
        st.success("✅ Answer Generated")
        st.markdown("#### 💡 Answer:")
        st.markdown(answer_text)
        
        # Step 3: Periodic Reflection
        st.markdown("### 🔍 Step 3: Quality Assessment")
        
        run_reflection = should_run_reflection()
        
        if run_reflection:
            st.markdown('<div class="reflection-info">⏰ <strong>Scheduled reflection time reached!</strong> Running quality assessment...</div>', unsafe_allow_html=True)
            
            with st.spinner("Evaluating answer quality..."):
                reflection_result = Reflector.evaluate_answer(user_query, answer_text, route)
            
            # Extract reflection text from CrewOutput
            if hasattr(reflection_result, 'raw'):
                reflection_text = reflection_result.raw
            elif hasattr(reflection_result, 'tasks_output') and reflection_result.tasks_output:
                reflection_text = reflection_result.tasks_output[0].raw
            else:
                reflection_text = str(reflection_result)
            
            # Parse quality score
            score_match = re.search(r'QUALITY SCORE:\s*(\d+)', reflection_text)
            quality_score = int(score_match.group(1)) if score_match else 0
            
            # Display quality metrics
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric("Quality Score", f"{quality_score}/100")
            
            # Display full reflection
            with st.expander("📊 Detailed Quality Assessment", expanded=True):
                st.markdown(reflection_text)
            
            # Update last reflection date
            st.session_state.last_reflection_date = datetime.now()
            st.success(f"✅ Reflection completed. Next reflection in {st.session_state.reflection_interval_days} days.")
            
        else:
            days_since_last = (datetime.now() - st.session_state.last_reflection_date).days if st.session_state.last_reflection_date else 0
            days_until_next = st.session_state.reflection_interval_days - days_since_last
            
            st.info(f"ℹ️ Quality assessment scheduled to run in {days_until_next} days. Use sidebar to trigger manually if needed.")
            
            # Show simplified quality indicator
            with st.expander("📈 Quick Quality Check"):
                st.markdown("""
                **Automated quality assessment is scheduled periodically.**
                
                Current Schedule: Every {interval} days  
                Last Assessment: {last_date}  
                Next Assessment: {next_date}
                
                You can manually trigger an assessment using the sidebar button.
                """.format(
                    interval=st.session_state.reflection_interval_days,
                    last_date=st.session_state.last_reflection_date.strftime('%Y-%m-%d') if st.session_state.last_reflection_date else "Not yet performed",
                    next_date=(st.session_state.last_reflection_date + timedelta(days=st.session_state.reflection_interval_days)).strftime('%Y-%m-%d') if st.session_state.last_reflection_date else "Soon"
                ))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Multi-Agent Math System</strong> | RAG-Powered | Mistral LLM | Periodic Quality Assessment</p>
</div>
""", unsafe_allow_html=True)