import autogen

# Accepted file formats for that can be stored in
# a vector database instance

from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func
from cerebrum.utils.communication import send_request


class AutoGenAgent:

    def __init__(self, agent_name):
        self.agent_name = agent_name

        # prepare autogen
        prepare_framework(FrameworkType.AutoGen)

    def run(self, task: str):
        """
        Convert currency amounts or show latest FX rates based on `task`.
        Examples of valid prompts:
            • "Convert 250 USD to EUR and JPY"
            • "How many GBP in 1000 CAD?"
            • "Show USD / EUR / GBP rates"

        Reply structure:
          ┌─────────────────────────────┐
          │ Assumptions (if any)        │
          │ Conversion Table            │
          │ Timestamp & Source          │
          └─────────────────────────────┘
        """
        # Hook Cerebrum ↔ AutoGen
        set_request_func(send_request, self.agent_name)

        # ── 1 ▸ CONVERTER ────────────────────────────────────────────────
        converter = autogen.AssistantAgent(
            name="converter",
            system_message=(
                "You are a currency-exchange assistant.  For ANY prompt, do NOT "
                "ask clarifying questions.  Instead:\n"
                "  • Parse amount, base currency, and target currencies.\n"
                "  • If any piece is missing, assume: amount=1, base='USD', "
                "    targets=['EUR', 'GBP', 'JPY'] and state that under "
                "    'Assumptions'.\n"
                "  • Write Python that calls the free API "
                "    https://api.exchangerate.host/latest.  No paid keys.\n"
                "  • After execution, format a neat table with converted values "
                "    rounded to 4 decimals.\n"
                "  • End with the timestamp (UTC) and API source.\n"
                "No follow-up questions."
            ),
            llm_config={"temperature": 0.3, "timeout": 600, "cache_seed": 42},
        )

        # ── 2 ▸ REVIEWER ─────────────────────────────────────────────────
        reviewer = autogen.AssistantAgent(
            name="reviewer",
            system_message=(
                "You are a QA reviewer.  Check the response for:\n"
                "  • Failed API calls / traceback\n"
                "  • Wrong currency codes or math errors\n"
                "  • Missing assumptions block when defaults are used\n"
                "If problems exist, list concise fixes; else reply 'APPROVED'."
            ),
            llm_config={"temperature": 0.2, "timeout": 300, "cache_seed": 99},
        )

        # ── 3 ▸ USER WRAPPER ────────────────────────────────────────────
        proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=6,
            code_execution_config={"use_docker": False, "work_dir": "fx_work"},
        )

        # A. generate conversion
        draft = proxy.initiate_chat(
            converter,
            message=task,
            summary_method="reflection_with_llm",
            max_turns=4,
        ).summary["content"]

        # B. review step
        audit = proxy.initiate_chat(
            reviewer,
            message=f"REPLY:\n{draft}\n---\nAudit please.",
            summary_method="reflection_with_llm",
            max_turns=2,
        ).summary["content"]

        # C. revise if needed
        if "APPROVED" not in audit.upper():
            final_reply = proxy.initiate_chat(
                converter,
                message=f"Please revise per these notes:\n{audit}",
                summary_method="reflection_with_llm",
                max_turns=1,
            ).summary["content"]
        else:
            final_reply = draft

        # ── 4 ▸ RETURN ──────────────────────────────────────────────────
        return {
            "agent_name": self.agent_name,
            "result": final_reply,
            "rounds": 1,   # Converter + Reviewer
        }





