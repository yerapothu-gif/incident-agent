import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage
import gradio as gr

from agent import get_agent_app

fastapi_app = FastAPI(title="Autonomous Incident Agent API")
agent_executor = get_agent_app()


class PromptRequest(BaseModel):
    thread_id: str
    message: str


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    rejection_reason: str | None = None

# --- FASTAPI REST ENDPOINTS ---

@fastapi_app.post("/chat")
async def chat_handler(request: PromptRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}

    agent_executor.invoke(inputs, config=config)
    snapshot = agent_executor.get_state(config)

    if snapshot.next and "sensitive_tools" in snapshot.next:
        pending_call = snapshot.values["messages"][-1].tool_calls[0]
        return {
            "thread_id": request.thread_id,
            "status": "AWAITING_APPROVAL",
            "pending_action": pending_call["name"],
            "parameters": pending_call["args"],
            "instruction": "Send POST to /approve to confirm or cancel this operation."
        }

    return {
        "thread_id": request.thread_id,
        "status": "COMPLETED",
        "response": snapshot.values["messages"][-1].content
    }


@fastapi_app.post("/approve")
async def approve_handler(request: ApprovalRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    snapshot = agent_executor.get_state(config)

    if not snapshot.next or "sensitive_tools" not in snapshot.next:
        raise HTTPException(status_code=400, detail="No pending sensitive action for this thread.")

    if request.approved:
        result = agent_executor.invoke(None, config=config)
        return {
            "thread_id": request.thread_id,
            "status": "RESOLVED",
            "response": result["messages"][-1].content
        }
    else:
        pending_call = snapshot.values["messages"][-1].tool_calls[0]
        rejection_msg = ToolMessage(
            tool_call_id=pending_call["id"],
            content=f"Rejected by engineer: {request.rejection_reason or 'No reason specified.'}"
        )
        agent_executor.update_state(config, {"messages": [rejection_msg]}, as_node="sensitive_tools")
        result = agent_executor.invoke(None, config=config)
        return {
            "thread_id": request.thread_id,
            "status": "REJECTED_AND_RESUMED",
            "response": result["messages"][-1].content
        }

# --- GRADIO INTERFACE LOGIC ---

def run_triage(thread_id: str, message: str):
    if not thread_id.strip() or not message.strip():
        return "Please provide both a Thread ID and an Incident Description.", ""

    config = {"configurable": {"thread_id": thread_id.strip()}}
    inputs = {"messages": [HumanMessage(content=message)]}

    agent_executor.invoke(inputs, config=config)
    snapshot = agent_executor.get_state(config)

    if snapshot.next and "sensitive_tools" in snapshot.next:
        pending_call = snapshot.values["messages"][-1].tool_calls[0]
        status_msg = (
            f"**ACTION REQUIRED: AWAITING APPROVAL**\n\n"
            f"- **Action:** `{pending_call['name']}`\n"
            f"- **Parameters:**\n```json\n{json.dumps(pending_call['args'], indent=2)}\n```\n\n"
            f"*Click 'Approve' or 'Reject' below to proceed.*"
        )
        return status_msg, "AWAITING_APPROVAL"

    return snapshot.values["messages"][-1].content, "COMPLETED"


def handle_decision(thread_id: str, decision: str, reason: str):
    if not thread_id.strip():
        return "Missing Thread ID.", ""

    config = {"configurable": {"thread_id": thread_id.strip()}}
    snapshot = agent_executor.get_state(config)

    if not snapshot.next or "sensitive_tools" not in snapshot.next:
        return "No pending sensitive action found for this Thread ID.", ""

    if decision == "Approve":
        result = agent_executor.invoke(None, config=config)
        return f"**Approved and executed:**\n\n{result['messages'][-1].content}", "RESOLVED"
    else:
        pending_call = snapshot.values["messages"][-1].tool_calls[0]
        rejection_msg = ToolMessage(
            tool_call_id=pending_call["id"],
            content=f"Rejected by engineer: {reason or 'Denied'}"
        )
        agent_executor.update_state(config, {"messages": [rejection_msg]}, as_node="sensitive_tools")
        result = agent_executor.invoke(None, config=config)
        return f"**Action rejected:**\n\n{result['messages'][-1].content}", "REJECTED"

# --- GRADIO UI LAYOUT ---

with gr.Blocks(title="Autonomous Incident Triage Agent") as demo:
    gr.Markdown("# Autonomous Incident Triage Agent")
    gr.Markdown("Investigates alerts, queries pgvector runbooks, and halts before critical operations.")

    with gr.Row():
        with gr.Column():
            thread_input = gr.Textbox(label="Thread ID", value="incident-001", placeholder="e.g. incident-001")
            prompt_input = gr.Textbox(
                label="Incident Description",
                lines=4,
                value="Auth service is timing out and runbook indicates manual intervention is required. Escalate immediately."
            )
            triage_btn = gr.Button("Trigger Triage", variant="primary")

            gr.Markdown("### Human-in-the-Loop Controls")
            reason_input = gr.Textbox(label="Rejection Reason (Optional)", placeholder="Only needed if rejecting")
            with gr.Row():
                approve_btn = gr.Button("Approve Escalation", variant="stop")
                reject_btn = gr.Button("Reject Action", variant="secondary")

        with gr.Column():
            status_output = gr.Label(label="Workflow State", value="READY")
            response_output = gr.Markdown(label="Agent Log / Output")

    triage_btn.click(
        fn=run_triage,
        inputs=[thread_input, prompt_input],
        outputs=[response_output, status_output]
    )
    approve_btn.click(
        fn=lambda tid: handle_decision(tid, "Approve", ""),
        inputs=[thread_input],
        outputs=[response_output, status_output]
    )
    reject_btn.click(
        fn=lambda tid, r: handle_decision(tid, "Reject", r),
        inputs=[thread_input, reason_input],
        outputs=[response_output, status_output]
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # Render injects PORT at runtime; 7860 is the local fallback.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))