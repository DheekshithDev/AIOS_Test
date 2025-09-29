from autogen import AssistantAgent, GroupChat, GroupChatManager

from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func
from cerebrum.utils.communication import send_request


class AutoGenAgent:

    def __init__(self, agent_name):
        self.agent_name = agent_name

        # prepare autogen
        prepare_framework(FrameworkType.AutoGen)

    def run(self, task):
        # set aios request function
        set_request_func(send_request, self.agent_name)

        # Create multiple AutoGen agents without user interaction
        research_agent = AssistantAgent(name="Researcher")  # Handles research on destinations
        itinerary_agent = AssistantAgent(name="ItineraryPlanner")  # Creates the itinerary
        budget_agent = AssistantAgent(name="BudgetAdvisor")  # Provides budget recommendations
        transport_agent = AssistantAgent(name="TransportAdvisor")  # Suggests transportation options

        # Set up group chat for collaboration
        groupchat = GroupChat(
            agents=[
                research_agent,
                itinerary_agent,
                budget_agent,
                transport_agent
            ],
            messages=[]  # Required parameter
        )
        manager = GroupChatManager(groupchat=groupchat)

        # Generate a structured message for the agents
        agent_message = (
            "Generate a comprehensive travel plan based on the following request: \n"
            f"{task}\n\n"
            "Provide:\n"
            "1. Destination research (attractions, best time to visit, culture).\n"
            "2. A detailed day-by-day itinerary with activities.\n"
            "3. Budget estimation, including accommodation, food, and activities.\n"
            "4. Transportation options (flights, local transport, car rentals).\n"
            "5. Additional travel tips and safety recommendations."
        )

        # Initiate conversation among AutoGen agents
        final_result = manager.initiate_chat(research_agent, message=agent_message, summary_method="reflection_with_llm", max_turns=3).summary
        return {
            "agent_name": self.agent_name,
            "result": final_result['content'],
            "rounds": 1,
        }

