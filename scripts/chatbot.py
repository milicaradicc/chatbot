import re
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

from config import Config
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
from data_fetcher import DataFetcher
from csv_data_fetcher import CSVDataFetcher
from milvus_handler import MilvusHandler
from openai_handler import OpenAIHandler
from sentence_transformers import SentenceTransformer

class ChatbotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot UI")
        self.root.geometry("600x400")

        # Initialize components
        self.embedder = SentenceTransformer(Config.SENTENCE_MODEL)
        self.data_fetcher = DataFetcher()
        self.csv_data_fetcher = CSVDataFetcher('data\data.csv')  # Load CSV
        self.milvus_handler = MilvusHandler(Config.MILVUS_HOST, Config.MILVUS_PORT, self.embedder)
        self.openai_handler = OpenAIHandler(Config.AZURE_ENDPOINT, Config.AZURE_API_KEY,Config.VERSION)

        # Fetch and process URLs
        self.sentences = self.data_fetcher.fetch_and_process_urls()
        self.csv_sentences = self.csv_data_fetcher.fetch_and_process_csv()  # Get CSV data

        # Combine web and CSV sentences
        self.sentences.extend(self.csv_sentences)

        self.embeddings = self.embedder.encode(self.sentences)

        # Set up Milvus
        self.milvus_handler.connect()
        self.milvus_handler.create_collection()
        self.milvus_handler.insert_embeddings(self.sentences, self.embeddings)

        # Create UI components
        self.create_widgets()

    def display_message(self, message):
        # Insert message into the chat history textbox
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + '\n')
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.yview(tk.END)  # Scroll to the bottom

    def query_csv_data(self, query: str, df: pd.DataFrame):
        """Function to query calendar data based on date or subject"""
        if 'date' in query.lower():
            date_str = re.search(r'\d{4}-\d{2}-\d{2}', query)  # Extract date (YYYY-MM-DD)
            if date_str:
                date = date_str.group(0)
                result = df[df['Start Date'] == date]
                if not result.empty:
                    return result[['Subject', 'Start Time', 'End Time', 'Location']].to_string(index=False)
                return "No events found on this date."
        elif 'subject' in query.lower():
            subject_match = re.search(r'subject\s*:\s*(\w+)', query, re.IGNORECASE)
            if subject_match:
                subject = subject_match.group(1)
                result = df[df['Subject'].str.contains(subject, case=False)]
                if not result.empty:
                    return result[['Subject', 'Start Date', 'Start Time', 'End Time', 'Location']].to_string(index=False)
                return f"No events found with the subject containing '{subject}'."
        return "I'm sorry, I couldn't understand your query related to the calendar."

    def interact_with_calendar_db(self, query: str):
        """Handles interaction with the calendar CSV data"""
        calendar_data = CSVDataFetcher('data\data.csv')  # Replace with your actual CSV path
        df = calendar_data.df
        response = self.query_csv_data(query, df)
        return response

    def interact_with_wikipedia_db(self, query: str):
        similar_sentences = self.milvus_handler.search_similar_sentences(query)
        if similar_sentences:
            return "\n".join(similar_sentences)
        else:
            return "No relevant information found from Wikipedia."

    def create_widgets(self):
        # Textbox for displaying chat history
        self.chat_history = scrolledtext.ScrolledText(self.root, height=15, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_history.grid(row=0, column=0, padx=10, pady=10)

        # Input field for user query
        self.query_input = tk.Entry(self.root, width=70)
        self.query_input.grid(row=1, column=0, padx=10, pady=10)

        # Button to send the query
        self.send_button = tk.Button(self.root, text="Send", width=10, command=self.handle_query)
        self.send_button.grid(row=2, column=0, padx=10, pady=10)

    def handle_query(self):
        query = self.query_input.get()
        if query.lower() == 'exit':
            self.root.quit()
            return

        self.display_message(f"You: {query}")

        similar_sentences = self.milvus_handler.search_similar_sentences(query)
        self.display_message("Similar Sentences:")
        for sentence in similar_sentences:
            self.display_message(f"- {sentence}")

        # Pass similar_sentences as context to generate_response
        response = self.openai_handler.generate_response(query, similar_sentences)
        if response:
            self.display_message(f"Response: {response}")
        else:
            self.display_message("No response generated.")

        self.query_input.delete(0, tk.END)

    def display_message(self, message):
        # Insert message into the chat history textbox
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + '\n')
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.yview(tk.END)  # Scroll to the bottom

# Main function to run the application
def main():
    try:
        # Create the Tkinter root window
        root = tk.Tk()
        chatbot_ui = ChatbotUI(root)
        root.mainloop()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
