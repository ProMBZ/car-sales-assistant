import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from tavily import TavilyClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure API Keys using os.environ
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GOOGLE_API_KEY or not TAVILY_API_KEY:
    st.error("API keys not found in .env file. Please add GOOGLE_API_KEY and TAVILY_API_KEY.")
    st.stop()

# Initialize Gemini 2.0 Flash LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Tavily Client
tavily_client = TavilyClient(TAVILY_API_KEY)

# Extensive Dummy Car Stock Data
car_stock = {
    "toyota corolla": {"price": 23000, "details": "2023 Toyota Corolla, excellent condition, low mileage, sunroof.", "benefits": "Reliable, fuel-efficient, perfect for commutes."},
    "honda vezel": {"price": 26000, "details": "2022 Honda Vezel, hybrid, well-maintained, navigation system.", "benefits": "Eco-friendly, spacious, advanced features."},
    "ford mustang": {"price": 35000, "details": "2021 Ford Mustang, sports edition, powerful engine, leather interior.", "benefits": "Performance-driven, stylish, thrilling to drive."},
    "nissan rogue": {"price": 28000, "details": "2023 Nissan Rogue, AWD, family-friendly, spacious cargo.", "benefits": "Safe, comfortable, ideal for road trips."},
    "chevrolet silverado": {"price": 40000, "details": "2020 Chevrolet Silverado, truck, heavy duty, tow package.", "benefits": "Powerful, durable, perfect for work or play."},
    "mercedes-benz c-class": {"price": 45000, "details": "2022 Mercedes-Benz C-Class, luxury sedan, premium sound, advanced safety.", "benefits": "Luxurious, refined, top-tier performance."},
    "bmw 3 series": {"price": 42000, "details": "2023 BMW 3 Series, sports sedan, dynamic handling, tech-packed.", "benefits": "Sporty, agile, cutting-edge technology."},
    "audi a4": {"price": 43000, "details": "2022 Audi A4, premium sedan, quattro AWD, virtual cockpit.", "benefits": "Elegant, all-weather capable, sophisticated design."},
    "volkswagen golf": {"price": 25000, "details": "2023 Volkswagen Golf, hatchback, sporty, fuel-efficient.", "benefits": "Practical, fun to drive, economical."},
    "hyundai tucson": {"price": 27000, "details": "2023 Hyundai Tucson, SUV, modern design, smart features.", "benefits": "Stylish, spacious, feature-rich."},
    "kia sportage": {"price": 26500, "details": "2022 Kia Sportage, SUV, reliable, comfortable ride.", "benefits": "Dependable, comfortable, value-packed."},
    "subaru outback": {"price": 30000, "details": "2023 Subaru Outback, AWD, adventure-ready, spacious interior.", "benefits": "Rugged, safe, perfect for outdoor enthusiasts."},
    "lexus rx": {"price": 50000, "details": "2022 Lexus RX, smooth ride, premium features.", "benefits": "Luxurious, comfortable, exceptional reliability."},
    "tesla model 3": {"price": 48000, "details": "2023 Tesla Model 3, electric sedan, autopilot, long range.", "benefits": "Electric, high-tech, environmentally friendly."},
    "porsche 911": {"price": 120000, "details": "2021 Porsche 911, sports car, high performance, iconic design.", "benefits": "High-performance, iconic, luxury sports car."},
    "jeep wrangler": {"price": 38000, "details": "2023 Jeep Wrangler, off-road, rugged, convertible.", "benefits": "Off-road capable, adventurous, iconic design."},
    "ram 1500": {"price": 42000, "details": "2022 Ram 1500, pickup truck, powerful, comfortable interior.", "benefits": "Powerful, versatile, comfortable for work or play."},
    "mini cooper": {"price": 24000, "details": "2023 Mini Cooper, compact, stylish, fun to drive.", "benefits": "Stylish, compact, fun and agile."},
    "land rover defender": {"price": 60000, "details": "2022 Land Rover Defender, off-road SUV, luxurious, robust.", "benefits": "Luxurious, off-road capable, robust and reliable."},
    "volvo xc90": {"price": 55000, "details": "2023 Volvo XC90, safest features, spacious.", "benefits": "Safe, spacious, luxurious and dependable."},
}

# Tavily Search Tool
def tavily_search_with_images(query):
    try:
        response = tavily_client.search(query=query, include_images=True)
        return response
    except Exception as e:
        return f"Error during Tavily search: {e}"

# Price Comparison (Enhanced)
def compare_prices(car_model):
    search_query = f"{car_model} price comparison"
    results = tavily_search_with_images(search_query)
    if isinstance(results, dict) and results.get('results'):
        competitor_price_info = results['results'][0]['content']
        search_url = results['results'][0]['url'] if results['results'] else "No search URL found."
        return f"Other dealers are selling the {car_model} at these prices: {competitor_price_info} Check it out here: [{search_url}]({search_url})"
    else:
        return "Could not retrieve competitor price information at this time."

# Car Details with Image Retrieval
def get_car_details(car_model):
    car_model_lower = car_model.lower()
    if car_model_lower in car_stock:
        details = car_stock[car_model_lower]
        search_results = tavily_search_with_images(f"{car_model} car")
        image_urls = []
        if isinstance(search_results, dict) and search_results.get('images'):
            image_urls = search_results['images']
        detail_string = (
            f"Absolutely! We have a {car_model.capitalize()} available. "
            f"Details: {details['details']}. "
            f"Price: ${details['price']:,}. "
            f"Benefits: {details['benefits']}."
        )
        return {"details": detail_string, "images": image_urls}
    else:
        return {"details": f"Sorry, the {car_model.capitalize()} is not currently in our stock.", "images": []}

# List Cars
def list_available_cars():
    if not car_stock:
        return "Our stock is currently empty."
    else:
        cars = ", ".join(car.capitalize() for car in car_stock)
        return f"We currently have the following cars in stock: {cars}."

# Collect Client Info
def collect_client_info():
    name = st.text_input("Please enter your name:")
    email = st.text_input("Please enter your email:")
    phone = st.text_input("Please enter your phone number:")
    if name and email and phone:
        return f"Client info: Name: {name}, Email: {email}, Phone: {phone}"
    else:
        return None
    # LangChain Agent Setup
tools = [
    Tool(name="ComparePrices", func=lambda car_model: compare_prices(car_model), description="Compares car prices with other dealers and provides links."),
    Tool(name="GetCarDetails", func=lambda car_model: get_car_details(car_model), description="Retrieves car details and images."),
    Tool(name="ListAvailableCars", func=lambda _: list_available_cars(), description="Lists all available cars in stock."),
    Tool(name="CollectClientInfo", func=collect_client_info, description="Collects client information for sales purposes.")
]

# Memory Initialization
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=st.session_state.memory)

# Streamlit UI - Modern Design
st.set_page_config(page_title="Car Sales Chatbot", layout="wide")

st.markdown(
    """
    <style>
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #121212; /* Dark background */
        color: #e0e0e0; /* Light text */
        margin: 0;
        padding: 0
    }
    .stApp {
        max-width: 1200px;
        margin: auto;
        padding: 2rem;
    }
    .st-eb {
        background-color: #1e1e1e; /* Darker card background */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .st-bb {
        background-color: #2a2a2a; /* Even darker message background */
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .st-bb:last-child {
        margin-bottom: 0;
    }
    .stTextInput>div>div>input {
        background-color: #333; /* Dark input background */
        border: 1px solid #555;
        border-radius: 5px;
        padding: 0.75rem;
        width: 100%;
        color: #e0e0e0; /* Light input text */
    }
    .stButton>button {
        background-color: #64b5f6; /* Blue button */
        color: #121212; /* Dark button text */
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stImage>img {
        border-radius: 8px;
        max-width: 100%;
        height: auto;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Car Sales Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I assist you with your car needs today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            results = agent.run(prompt)
            if isinstance(results, dict) and 'images' in results:
                full_response = results['details']
                message_placeholder.markdown(full_response)
                if results['images']:
                    for image_url in results['images']:
                        st.image(image_url, caption="Search Result Image", use_column_width=True)
            else:
                full_response = str(results)
                message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})