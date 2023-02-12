from urllib.request import urlopen
from bs4 import BeautifulSoup


def clean_text(txt):
    return txt.strip().replace('  ', '').replace('\n\n', '\n').replace('\n\n', '\n')


def find_title_abstract(pubmed_id):
    url = f'https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/'
    page = urlopen(url)
    html = page.read().decode('utf-8')
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.find("title").text)
    abstract = clean_text(
        soup.find("div", {"class": "abstract-content selected"}).text)
    return (title, abstract)
    
