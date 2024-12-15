import requests
from bs4 import BeautifulSoup
import re
from typing import List

class DataFetcher:
    def __init__(self, path: str = "data/links"):
        self.path = path
    
    def fetch_and_process_urls(self) -> List[str]:
        try:
            with open(self.path, 'r') as f:
                urls = f.read().splitlines()
        except FileNotFoundError:
            # default URLs if no links file found
            urls = [
                "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "https://en.wikipedia.org/wiki/Artificial_intelligence"
            ]
        
        urls = [url.strip().rstrip(',') for url in urls if url.strip()]
        all_sentences = []
        for url in urls:
            sentences = self.fetch_and_split_sentences(url)
            all_sentences.extend(sentences)
        return all_sentences

    def fetch_and_split_sentences(self, url: str) -> List[str]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            sentences = []
            paragraphs = soup.find_all('p')
            for paragraph in paragraphs:
                text = ' '.join(paragraph.get_text().split())
                text = re.sub(r'&[a-z]+;', '', text)

                paragraph_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
                sentences.extend([s.strip() + '.' for s in paragraph_sentences if s.strip() and len(s) > 10])

            return sentences

        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return []
