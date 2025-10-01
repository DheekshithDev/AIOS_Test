from autogen import ConversableAgent
import numpy as np
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from cerebrum.community.adapter import prepare_framework, FrameworkType, set_request_func
from cerebrum.utils.communication import send_request
import torch
torch.cuda.empty_cache()

class AutoGenAgent:

    def __init__(self, agent_name):
        self.agent_name = agent_name

        # prepare autogen
        prepare_framework(FrameworkType.AutoGen)

    def run(self, task):
        # set aios request function
        set_request_func(send_request, self.agent_name)

        # ===== Configuration =====
        SUPPORT_SIZE = 5  # Fixed support set size
        TOP_K = 3  # Return top 3 recommended agents

        # ===== Initial Support Query Set =====
        agents = {
            'WebSearch': [
                "Find the latest news about AI regulation in the European Union.",
                "Search for upcoming tech conferences in North America in 2025.",
                "Look up reviews for the newest electric cars released this year.",
                "Find current job openings for data scientists in New York City.",
                "Search for tutorials on how to deploy a Flask app to AWS."
            ],
            'SoftwareProductIdea': [
                "Generate software product ideas for improving remote team collaboration in small startups.",
                "Suggest innovative app concepts that help users reduce screen time and improve digital well-being.",
                "Come up with software product ideas that assist students in managing coursework and deadlines.",
                "Propose new SaaS product ideas targeting independent e-commerce sellers.",
                "Generate ideas for AI-powered tools to support content creators in planning and editing their videos."
            ],
            'DataVisualizetion': [
                "Generate and visualize the distribution of ages in a synthetic dataset using a histogram.",
                "Create a line chart showing monthly sales trends from a given CSV file.",
                "Write code to visualize the correlation matrix of a dataset using a heatmap.",
                "Plot a pie chart of user device usage statistics from a JSON data source.",
                "Build a bar graph comparing average temperatures across major cities using real-time weather data."
            ],
            'ResearchHelper': [
                "Find recent studies on the impact of remote work on employee productivity.",
                "Research the latest advancements in quantum computing.",
                "Look up historical trends in electric vehicle adoption worldwide.",
                "Find sources discussing the ethical concerns of facial recognition technology.",
                "Research current methods used in detecting fake news online."
            ],
            'TaskSolving': [
                "Create a Python script that visualizes COVID-19 trends using the latest data from an online source.",
                "Build a tool that fetches real-time cryptocurrency prices and plots them using matplotlib.",
                "Write a program that scrapes job listings for data science roles and summarizes common skill requirements.",
                "Develop a script to pull weather forecasts from an online API and display them in a weekly calendar format.",
                "Generate a report on top trending GitHub repositories by language using web scraping and data visualization."
            ],
            'Research': [
                "Retrieve and summarize the latest papers on federated learning in healthcare. I’ll approve the plan before moving forward.",
                "Help me build a research pipeline to explore self-supervised learning methods in natural language processing. Wait for my approval after the plan.",
                "Find academic papers about reinforcement learning in robotics and summarize key innovations. I’ll review and approve each stage.",
                "Create a research workflow to study bias mitigation techniques in LLMs. Pause after planning until I type 'Approve'.",
                "Retrieve recent publications on graph neural networks for recommendation systems. Let me review the summary before finalizing."
            ],
            'TasksSolving_Nested': [
                "Plan a birthday event and consult other agents for venue suggestions, budget estimation, and invitation design.",
                "Organize a research project workflow by coordinating with agents specialized in literature review, data collection, and analysis.",
                "Solve a user’s coding issue by using one agent to analyze the error and another to rewrite the buggy function.",
                "Create a detailed travel itinerary by delegating destination research, budgeting, and daily activity planning to different agents.",
                "Assist with healthy meal planning by asking one agent to analyze dietary preferences and another to generate recipe suggestions."
            ],
            'summary_agent': [
                "Summarize this conversation where the user discusses planning a vacation to Japan with an AI travel assistant.",
                "Generate a concise summary of a support chat where a user troubleshoots login issues with a chatbot.",
                "Create a summary of a student’s conversation with an AI tutor about preparing for a calculus exam.",
                "Summarize a mental health chatbot interaction focusing on emotional context and key coping strategies suggested.",
                "Provide a brief summary of a multi-turn chat where the user explores starting a small business with help from an AI advisor."
            ],
            'CurrencyCalculator': [
                "Convert 100 USD to EUR.",
                "How much is 50 EUR in USD?",
                "What’s the equivalent of 200 USD in Euros?",
                "Convert 75 Euros to U.S. dollars.",
                "I want to exchange 300 USD—how many Euros will I get?"
            ],
            'MathProblem': [
                "What is 25 divided by 5?",
                "Solve: 7 + 8 × 2.",
                "What is the square root of 81?",
                "Find the value of x if x + 4 = 10.",
                "What is the perimeter of a rectangle with length 6 and width 3?"
            ],
            'PlanningAndSolving': [
                "Plan a step-by-step approach to learn Python for data analysis in two months.",
                "Devise a multi-stage strategy to prepare for a machine learning job interview.",
                "Outline a plan to reduce personal expenses by 30% over the next three months.",
                "Break down the process of writing and publishing a research paper in a top-tier AI conference.",
                "Develop a strategy to troubleshoot and fix intermittent Wi-Fi connectivity issues at home."
            ],
            'generation_agent': [
                "Generate a JSON object describing a user's profile including name, age, and interests.",
                "Create JSON output for a product catalog item with fields: product_id, name, price, and availability.",
                "Return a JSON response that models a booking confirmation with fields: booking_id, user_info, and itinerary.",
                "Output JSON representing a blog post structure with title, author, tags, and content sections.",
                "Generate JSON data for a chatbot response containing intent, confidence score, and response text."
            ],
            'Code_GenerationAndDebugging': [
                "Write and debug a Python script that scrapes product prices from an e-commerce website.",
                "Generate and execute code to calculate the eigenvalues of a given matrix using NumPy.",
                "Create a function to detect palindromes in a list of strings and fix any runtime errors.",
                "Develop and debug a script that fetches live weather data from an API and visualizes it.",
                "Implement and test a sorting algorithm for large datasets, identifying and fixing performance bottlenecks."
            ],
            'academic_agent': [
                "Find and summarize recent research papers on contrastive learning in computer vision.",
                "Help me generate research questions related to the ethical implications of AI in healthcare.",
                "Summarize the key findings from the latest studies on climate change and urban planning.",
                "Retrieve academic literature on reinforcement learning applications in robotics.",
                "Suggest potential research directions based on gaps in the literature about memory mechanisms in LLMs."
            ],
            'travel_planner_agent': [
                "Plan a 5-day budget-friendly trip to Kyoto focused on cultural experiences and local cuisine.",
                "Create a romantic weekend itinerary for Paris with a mid-range budget and emphasis on art and wine.",
                "Suggest a solo adventure itinerary in New Zealand for nature hikes and photography in early spring.",
                "Generate a family-friendly travel plan to Orlando that balances theme parks and relaxing downtime.",
                "Design a luxury beach vacation itinerary in the Maldives for a honeymoon in late November."
            ],
            'healthy_agent': [
                "Analyze my weekly diet and suggest personalized improvements based on heart health.",
                "Assess my health risks given my recent blood test results and fitness tracker data.",
                "Retrieve medical research on the effects of intermittent fasting on metabolism.",
                "Generate a personalized sleep improvement plan using my sleep tracking data.",
                "Recommend lifestyle changes for managing prediabetes based on my age and BMI."
            ],
            'code_translate_agent': [
                "Analyze this Python function for data preprocessing and generate clear Markdown documentation.",
                "Generate concise Markdown docs for a Python class handling API requests and response parsing.",
                "Translate a Python script for training a neural network into step-by-step Markdown explanations.",
                "Document a Python module that performs web scraping and data cleaning using BeautifulSoup and pandas.",
                "Create readable Markdown documentation for a multi-file Python project implementing a recommendation system."
            ],
            'writting_agent': [
                "Write a story using the keywords: time machine, forgotten diary, and eclipse.",
                "Create a short tale based on the keywords: pirate ship, lost treasure, and stormy sea.",
                "Generate a story using the keywords: artificial intelligence, rebellion, and empathy.",
                "Write a fantasy story with the keywords: dragon egg, forest guardian, and ancient spell.",
                "Craft a mystery using the keywords: locked room, missing painting, and midnight call."
            ],
            'cocktail_mixlogist': [
                "Create a refreshing summer mocktail using watermelon, mint, and lime.",
                "Suggest a low-sugar cocktail for a keto-friendly dinner party.",
                "Make a cozy winter drink featuring bourbon, cinnamon, and apple cider.",
                "Design a tropical cocktail using rum, pineapple, and coconut cream for a beach-themed event.",
                "Recommend a non-alcoholic cocktail that pairs well with spicy Indian food."
            ],
            'creation_agent': [
                "Create an Instagram post promoting a new vegan restaurant with catchy text and a vibrant image.",
                "Design a motivational quote graphic with an inspiring caption for LinkedIn.",
                "Generate a Twitter thread teaser for an upcoming YouTube tech review video.",
                "Write an engaging caption and design a lifestyle image for a beachwear brand on Facebook.",
                "Produce a carousel post with tips for remote productivity, including text and visuals."
            ],
            'language_tutor': [
                "Help me practice daily conversation phrases in French for my upcoming trip to Paris.",
                "Explain the difference between past simple and present perfect in English with examples.",
                "Provide vocabulary exercises focused on food and dining in Japanese.",
                "Guide me through proper pronunciation of tricky Spanish consonants like 'rr' and 'ñ'.",
                "Teach me cultural etiquette and useful greetings for business meetings in South Korea."
            ],
            'logo_creator': [
                "Design a sleek and minimal logo for a fintech startup targeting Gen Z.",
                "Create a playful and colorful logo for a children’s educational toy brand.",
                "Generate a vintage-style logo for a local artisanal coffee shop.",
                "Design a bold and modern logo for an esports team named 'ShadowStrike'.",
                "Create a nature-inspired logo for a sustainable skincare company."
            ],
            'math_agent': [
                "Solve a complex integral and explain each step clearly.",
                "Find and summarize recent research on graph coloring algorithms.",
                "Help me understand the proof of the Cauchy-Schwarz inequality.",
                "Provide a step-by-step solution to a real-world optimization problem using linear programming.",
                "Search for academic papers discussing recent advances in number theory related to prime distribution."
            ],
            'meme_creator': [
                "Make a meme about the struggles of working from home with pets.",
                "Create a meme using an image of a confused cat reacting to math homework.",
                "Generate a meme about the endless cycle of opening and closing the fridge.",
                "Design a funny meme related to AI taking over mundane office tasks.",
                "Craft a meme using the 'Distracted Boyfriend' template to show procrastination vs productivity."
            ],
            'music_composer': [
                "Compose a calming piano piece suitable for meditation sessions.",
                "Create an upbeat pop track for a summer travel vlog.",
                "Generate a suspenseful orchestral score for a short horror film.",
                "Produce a lo-fi background track for a study and focus playlist.",
                "Write a romantic acoustic melody for a wedding ceremony entrance."
            ],
            'story_teller': [
                "Tell a fantasy story set in a floating city ruled by sentient birds.",
                "Create a sci-fi thriller featuring a time-traveling detective and an ancient AI.",
                "Craft a bedtime story for kids about a curious panda exploring a magical bamboo forest.",
                "Narrate a historical tale set during the Renaissance with vivid audio descriptions.",
                "Generate a romantic short story based on two characters who meet during a lunar eclipse."
            ],
            'tech_support_agent': [
                "I need help troubleshooting my computer.",
                "How can I update my software efficiently?",
                "Recommend the best diagnostic tools for my system.",
                "What are the latest updates in technical support?",
                "Help me resolve connectivity issues."
            ],
            'festival_card_designer': [
                "Design a Diwali card that blends traditional motifs with a modern minimalist aesthetic.",
                "Create a Christmas greeting card for a tech-savvy audience with a playful tone.",
                "Generate a personalized New Year card with a travel theme for a family of four.",
                "Design a colorful Holi card that incorporates watercolor effects and festive slogans.",
                "Create a romantic Valentine’s Day card tailored for a long-distance couple."
            ],
        }

        agent_names = list(agents.keys())

        # ===== Embedding using all-mpnet-base-v2 =====
        embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

        def embed_text(text):
            return embedding_model.encode(text)

        embedding_dim = len(embed_text("example query"))

        # ===== AgentPrototypes: store each support entry as (query_text, embedding) =====
        class AgentPrototypes:
            def __init__(self, max_support=SUPPORT_SIZE):
                self.max_support = max_support
                self.support_entries = {agent: deque(maxlen=max_support) for agent in agent_names}

            def add_entry(self, agent, query, emb):
                self.support_entries[agent].append((query, emb))

            def compute_prototype(self, agent):
                entries = self.support_entries[agent]
                if entries:
                    embs = np.vstack([emb for (q, emb) in entries])
                    return np.mean(embs, axis=0)
                else:
                    return np.zeros(embedding_dim)

            def compute_all_prototypes(self):
                return {agent: self.compute_prototype(agent) for agent in agent_names}

        def populate_prototypes(proto_obj):
            for agent, query_list in agents.items():
                for q in query_list:
                    emb = embed_text(q)
                    proto_obj.add_entry(agent, q, emb)

        # ===== Recommendation Function =====
        def recommend_agents(query, proto_obj, top_k=TOP_K):
            query_emb = embed_text(query)
            prototypes = proto_obj.compute_all_prototypes()
            similarities = {}
            for agent, proto in prototypes.items():
                sim = cosine_similarity(query_emb.reshape(1, -1), proto.reshape(1, -1))[0][0]
                similarities[agent] = sim
            sorted_agents = sorted(similarities, key=similarities.get, reverse=True)
            return sorted_agents[:top_k]

        # ===== Active Learning Update (Euclidean-based selection) =====
        def active_learning_update_euclidean(proto_obj, agent, new_query, embed, top_k=SUPPORT_SIZE):
            new_emb = embed(new_query)
            current = list(proto_obj.support_entries[agent])
            candidates = current + [(new_query, new_emb)]
            if current:
                current_proto = np.mean(np.vstack([emb for (q, emb) in current]), axis=0)
            else:
                current_proto = np.zeros_like(new_emb)
            distances = np.linalg.norm(np.vstack([emb for (q, emb) in candidates]) - current_proto, axis=1)
            indices = np.argsort(distances)[-top_k:]
            new_set = [candidates[i] for i in indices]
            proto_obj.support_entries[agent].clear()
            for entry in new_set:
                proto_obj.support_entries[agent].append(entry)

        # ===== Final Function: Recommend and Update =====
        def recommend_and_update(query, proto_obj):
            # Recommend agents based on current prototypes.
            recommended = recommend_agents(query, proto_obj, TOP_K)
            # Update dynamic prototypes for the top recommended agent.
            top_agent = recommended[0]
            active_learning_update_euclidean(proto_obj, top_agent, query, embed_text)
            return ", ".join(recommended)

        dynamic_proto = AgentPrototypes()
        populate_prototypes(dynamic_proto)

        rec_agents = recommend_and_update(task, dynamic_proto)

        return {
            "agent_name": self.agent_name,
            "result": rec_agents,
            "rounds": 1,
        }
