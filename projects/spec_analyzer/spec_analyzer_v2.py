import os
import csv
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# --- CONFIGURATION (Same as before) ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
gemini_model = ChatVertexAI(
    model_name="gemini-2.5-flash",  # or "gemini-2.0-flash-exp"
    temperature=0,
    max_output_tokens=2048,
    project=GCP_PROJECT_ID,
    location="us-central1"
)

# If you don't have Azure yet, you can swap this for standard ChatOpenAI
azure_model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
    openai_api_version=os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT",
                             "https://your-org.openai.azure.com/"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
# --- THE STATE ---
# We add 'human_feedback' so the AI knows what you want changed


class AgentState(TypedDict):
    spec_text: str
    analysis_gaps: str
    ado_tickets: str
    human_feedback: str

# --- NODES ---


def analyze_requirements_node(state: AgentState):
    print("\n--- STEP 1: Google Gemini is analyzing (or re-analyzing)... ---")

    # Check if there is feedback from a previous loop
    feedback_context = ""
    if state.get('human_feedback'):
        print(f"   (Incorporating feedback: {state['human_feedback']})")
        feedback_context = f"\n\nIMPORTANT: The user rejected the previous draft. Fix it based on this feedback: {state['human_feedback']}"

    prompt = f"""
    You are a Senior Solutions Architect. Analyze this requirement text.
    Identify gaps and missing criteria.
    {feedback_context}
    
    REQUIREMENT TEXT:
    {state['spec_text']}
    """
    response = gemini_model.invoke([HumanMessage(content=prompt)])
    return {"analysis_gaps": response.content}


def draft_tickets_node(state: AgentState):
    print("\n--- STEP 2: Azure OpenAI is drafting Azure DevOps tickets... ---")

    analysis = state['analysis_gaps']
    original_spec = state['spec_text']

    prompt = f"""
    You are a Technical Product Owner.
    Based on the original request and the Architect's gap analysis below,
    write 3-5 structured Azure DevOps Work Items.

    Return ONLY a valid JSON array. Each work item must have these exact fields:
    - "Work Item Type": Must be one of: "User Story", "Task", "Bug", "Feature"
    - "Title": A clear, concise title (max 100 characters)
    - "Description": Detailed description of the work item
    - "Acceptance Criteria": Bulleted list of criteria (use "- " for bullets)
    - "Priority": Must be one of: "1", "2", "3", "4"

    ORIGINAL REQUEST:
    {original_spec}

    ARCHITECT'S ANALYSIS:
    {analysis}

    Return ONLY the JSON array, no additional text or markdown code blocks.
    """
    response = azure_model.invoke([HumanMessage(content=prompt)])
    return {"ado_tickets": response.content}


def human_review_node(state: AgentState):
    print("\n--- STEP 3: HUMAN REVIEW REQUIRED ---")
    print("Current Draft Tickets:\n")
    print(state['ado_tickets'])

    # This pauses the graph! The state is saved to memory.
    # The user will input data, and the graph resumes with that data.
    review_decision = interrupt(
        "Please review the tickets. Type 'approve' to save, or type your feedback to fix them."
    )

    # We return the decision so the conditional edge can see it
    return {"human_feedback": review_decision}


def export_to_ado_csv(tickets_json: str, output_file: str = "ado_work_items.csv"):
    """
    Exports the JSON tickets to a CSV file compatible with Azure DevOps import.

    Args:
        tickets_json: JSON string containing the work items
        output_file: Path to the output CSV file

    Returns:
        Path to the created CSV file
    """
    try:
        # Clean up the JSON string (remove markdown code blocks if present)
        cleaned_json = tickets_json.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json.split("```json")[1]
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json.split("```")[1]
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json.rsplit("```", 1)[0]
        cleaned_json = cleaned_json.strip()

        # Parse the JSON
        tickets = json.loads(cleaned_json)

        # Azure DevOps CSV format headers
        fieldnames = [
            "Work Item Type",
            "Title",
            "Description",
            "Acceptance Criteria",
            "Priority"
        ]

        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ticket in tickets:
                writer.writerow({
                    "Work Item Type": ticket.get("Work Item Type", "User Story"),
                    "Title": ticket.get("Title", ""),
                    "Description": ticket.get("Description", ""),
                    "Acceptance Criteria": ticket.get("Acceptance Criteria", ""),
                    "Priority": ticket.get("Priority", "2")
                })

        print(
            f"\n✓ Successfully exported {len(tickets)} work items to: {output_file}")
        return output_file

    except json.JSONDecodeError as e:
        print(f"\n✗ Error parsing JSON: {e}")
        print(f"Response content:\n{tickets_json}")
        # Fallback: Save the raw output to a text file
        fallback_file = "ado_work_items_raw.txt"
        with open(fallback_file, 'w', encoding='utf-8') as f:
            f.write(tickets_json)
        print(f"Saved raw output to: {fallback_file}")
        return None
    except Exception as e:
        print(f"\n✗ Error exporting to CSV: {e}")
        return None


def save_to_file_node(state: AgentState):
    print("\n--- STEP 4: Saving to Disk ---")
    json_filename = "ado_work_items.json"

    # Clean up the markdown code blocks if the LLM added them
    clean_json = state['ado_tickets'].replace(
        "```json", "").replace("```", "")

    # Save JSON file
    with open(json_filename, "w") as f:
        f.write(clean_json)

    print(f"✅ Success! Tickets saved to {os.path.abspath(json_filename)}")

    # Export to CSV for Azure DevOps import
    csv_file = export_to_ado_csv(state['ado_tickets'])

    if csv_file:
        print(f"\n📋 Import Instructions:")
        print(f"   1. Go to Azure DevOps > Boards > Work Items")
        print(f"   2. Click 'Import Work Items' or use the Excel plugin")
        print(f"   3. Upload the file: {csv_file}")
        print(f"   4. Map the columns if prompted")
        print(f"   5. Complete the import")

    return {}

# --- ROUTING LOGIC ---


def router(state: AgentState) -> Literal["analyze_spec", "save_tickets"]:
    # Logic: If the user typed "approve", save it. Otherwise, loop back.
    decision = state.get("human_feedback", "").lower()

    if decision == "approve":
        return "save_tickets"
    else:
        print(f"\n🔄 Loop: Sending feedback back to Architect...")
        return "analyze_spec"


# --- THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("analyze_spec", analyze_requirements_node)
workflow.add_node("create_tickets", draft_tickets_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("save_tickets", save_to_file_node)

# Build the flow
workflow.set_entry_point("analyze_spec")
workflow.add_edge("analyze_spec", "create_tickets")
workflow.add_edge("create_tickets", "human_review")

# The Conditional Edge is the "Brain" of the routing
workflow.add_conditional_edges(
    "human_review",
    router
)
workflow.add_edge("save_tickets", END)

# MemorySaver is required for 'interrupt' to work
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # We need a thread_id to track this specific "conversation"
    config = {"configurable": {"thread_id": "ticket-session-1"}}

    initial_spec = "Build a login page for the mobile app."

    print(f"INPUT: {initial_spec}")

    # FIRST RUN: It will run until it hits the interrupt
    result = app.invoke({"spec_text": initial_spec}, config=config)

    # THE INTERRUPT HANDLER
    # When the code reaches here, it has PAUSED inside 'human_review_node'.
    # We now act as the "Frontend" getting user input.

    while True:
        # Check if we are currently interrupted
        snapshot = app.get_state(config)
        if not snapshot.next:
            break  # Graph finished

        # Ask the user (You) for input
        user_input = input(
            "\n[Manager] Your Decision (approve / [feedback]): ")

        # RESUME the graph with the user's input
        # The Command object tells LangGraph "Here is the result of the interrupt, keep going"
        result = app.invoke(
            Command(resume=user_input),
            config=config
        )
