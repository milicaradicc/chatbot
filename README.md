# AI-Powered Context-Aware Chatbot

## 🤖 Project Overview
![Chatbot Interface](images\image.png)
This is an intelligent chatbot application that leverages cutting-edge AI technologies to provide context-aware responses. The chatbot uses web scraping, sentence embeddings, vector search, and OpenAI's language models to generate intelligent and contextually relevant answers.

## ✨ Key Features

- **Web Content Extraction**: Automatically fetches and processes text from specified URLs
- **Semantic Search**: Uses Sentence Transformers to create embeddings
- **Vector Database**: Employs Milvus for efficient similarity search
- **AI-Powered Responses**: Generates answers using Azure OpenAI's language models
- **User-Friendly Interface**: Tkinter-based GUI for easy interaction

## 🛠 Technologies Used

- Python
- Sentence Transformers
- Milvus Vector Database
- Azure OpenAI
- BeautifulSoup
- Tkinter

## 📦 Prerequisites

- Python 3.8+
- Azure OpenAI API Access
- Milvus Vector Database

## 🚀 Installation

1. Clone the repository
   ```bash
   git clone https://github.com/milicaradicc/chatbot
   cd chatbot
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   Create a `.env` file with the following:
   ```
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   AZURE_ENDPOINT=your_azure_endpoint
   AZURE_API_KEY=your_azure_api_key
   SENTENCE_MODEL=all-MiniLM-L6-v2
   ```

## 🖥 Running the Application

```bash
python main.py
```

## 📞 Contact

Milica Radic - milica.t.radic@gmail.com

Project Link: [https://github.com/milicaradicc/chatbot](https://github.com/milicaradicc/chatbot)

