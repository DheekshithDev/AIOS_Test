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
        Generate a structured interview kit for the role described by `task`.
        The kit contains:
          • Role snapshot & assumed details (if any data are missing)
          • 12–15 questions in four categories:
              – Technical / Domain
              – Behavioural / Situational
              – Problem-solving / Case
              – Culture & Values
          • For each question: competency tested + sample good answer points
          • A simple 1-5 scoring rubric

        No follow-up questions: missing info is filled with explicit assumptions.
        """
        # Glue Cerebrum ↔ AutoGen
        set_request_func(send_request, self.agent_name)

        # ── 1 ▸ QUESTION DESIGNER ───────────────────────────────────────────
        designer = autogen.AssistantAgent(
            name="designer",
            system_message=(
                "You are a senior HR specialist and hiring manager.\n"
                "Upon any prompt, ALWAYS create a full interview kit:\n"
                "  • *Assumptions* block listing default salary band, seniority, "
                "    or tech stack if not specified.\n"
                "  • 12–15 questions across 4 categories, each with:\n"
                "      – Competency/skill assessed\n"
                "      – Bullet 'What a strong answer shows'\n"
                "  • End with a generic 1-to-5 scoring rubric.\n"
                "Never ask the candidate (user) for more information."
            ),
            llm_config={"temperature": 0.65, "timeout": 600, "cache_seed": 42},
        )

        # ── 2 ▸ QUALITY REVIEWER ──────────────────────────────────────────
        reviewer = autogen.AssistantAgent(
            name="reviewer",
            system_message=(
                "You are an interview-design auditor.  Check the kit for:\n"
                "  • Redundancy or bias\n"
                "  • Illegal or discriminatory questions\n"
                "  • Inadequate coverage of role competencies\n"
                "If issues exist, return concise correction notes; else reply 'APPROVED'."
            ),
            llm_config={"temperature": 0.25, "timeout": 300, "cache_seed": 99},
        )

        # ── 3 ▸ USER PROXY WRAPPER ─────────────────────────────────────────
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=6,
        )

        # A. draft interview kit
        draft_kit = user_proxy.initiate_chat(
            designer,
            message=task,
            summary_method="reflection_with_llm",
            max_turns=3,
        ).summary["content"]

        # B. reviewer audit
        audit_notes = user_proxy.initiate_chat(
            reviewer,
            message=f"KIT:\n{draft_kit}\n---\nAudit this kit.",
            summary_method="reflection_with_llm",
            max_turns=2,
        ).summary["content"]

        # C. revise if needed
        if "APPROVED" not in audit_notes.upper():
            final_kit = user_proxy.initiate_chat(
                designer,
                message=f"Revise per these notes:\n{audit_notes}",
                summary_method="last_msg",
                max_turns=1,
            ).summary
        else:
            final_kit = draft_kit

        # ── 4 ▸ RETURN ────────────────────────────────────────────────────
        return {
            "agent_name": self.agent_name,
            "result": final_kit,
            "rounds": 1,   # Designer + Reviewer
        }



