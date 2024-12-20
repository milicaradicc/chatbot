import re
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

from config import Config
import pandas as pd
from data_fetcher import DataFetcher
from csv_data_fetcher import CSVDataFetcher
from milvus_handler import MilvusHandler
from openai_handler import OpenAIHandler
from sentence_transformers import SentenceTransformer

import tkinter as tk
from tkinter import scrolledtext

# Load environment variables
load_dotenv()

class Chatbot:
    def __init__(self):
        # Initialize components
        self.embedder = SentenceTransformer(Config.SENTENCE_MODEL)
        self.data_fetcher = DataFetcher()
        self.csv_data_fetcher = CSVDataFetcher('data/data.csv')  # Load CSV
        self.milvus_handler = MilvusHandler(Config.MILVUS_HOST, Config.MILVUS_PORT, self.embedder)
        self.openai_handler = OpenAIHandler(Config.AZURE_ENDPOINT, Config.AZURE_API_KEY, Config.VERSION, "data/data.csv")

        # Fetch and process URLs and CSV data
        self.sentences = self.data_fetcher.fetch_and_process_urls()
        self.csv_sentences = self.csv_data_fetcher.fetch_and_process_csv()

        # Combine web and CSV sentences
        self.sentences.extend(self.csv_sentences)

        self.embeddings = self.embedder.encode(self.sentences)

        # Set up Milvus
        self.milvus_handler.connect()
        self.milvus_handler.create_collection()
        self.milvus_handler.insert_embeddings(self.sentences, self.embeddings)

    def get_similar_sentences(self, query: str):
        return self.milvus_handler.search_similar_sentences(query)

    def generate_response(self, query: str, similar_sentences):
        return self.openai_handler.generate_response(query, similar_sentences)

class ChatUI:
    def __init__(self, root, chatbot):
        self.chatbot = chatbot

        self.root = root
        self.root.title("Chatbot Interface")
        
        self.frame = tk.Frame(root)
        self.frame.pack(padx=10, pady=10)

        self.chat_history = scrolledtext.ScrolledText(self.frame, height=20, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.chat_history.tag_config("user", foreground="blue")  # User messages in blue
        self.chat_history.tag_config("bot", foreground="green")  # Bot messages in green

        self.user_input = tk.Entry(self.frame, width=60)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        self.submit_button = tk.Button(self.frame, text="Send", command=self.handle_query)
        self.submit_button.grid(row=1, column=1, padx=10, pady=10)

        self.similar_sentences_text = tk.Text(self.frame, height=10, width=40, wrap=tk.WORD, state=tk.DISABLED)
        self.similar_sentences_text.grid(row=0, column=2, rowspan=2, padx=10, pady=10)

    def handle_query(self):
        query = self.user_input.get()
        if query.strip() == '':
            return

        self.display_message(query, "User")

        similar_sentences = self.chatbot.get_similar_sentences(query)

        self.display_similar_sentences(similar_sentences)

        response = self.chatbot.generate_response(query, similar_sentences)

        self.display_message(response, "Bot")

        self.user_input.delete(0, tk.END)

    def display_message(self, message, sender):
        self.chat_history.config(state=tk.NORMAL)
        if sender == "User":
            self.chat_history.insert(tk.END, f"You: {message}\n", "user")
        else:
            self.chat_history.insert(tk.END, f"Bot: {message}\n", "bot")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.yview(tk.END)

    def display_similar_sentences(self, similar_sentences):
        self.similar_sentences_text.config(state=tk.NORMAL)
        self.similar_sentences_text.delete(1.0, tk.END)
        if similar_sentences:
            self.similar_sentences_text.insert(tk.END, "Similar sentences:\n")
            for sentence in similar_sentences:
                self.similar_sentences_text.insert(tk.END, f"- {sentence}\n")
        else:
            self.similar_sentences_text.insert(tk.END, "No similar sentences found.")
        self.similar_sentences_text.config(state=tk.DISABLED)

def main():
    chatbot = Chatbot()
    root = tk.Tk()
    chat_ui = ChatUI(root, chatbot)
    root.mainloop()

if __name__ == "__main__":
    main()
