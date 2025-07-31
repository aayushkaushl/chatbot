import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=' ')
    return ' '.join(text.split())
