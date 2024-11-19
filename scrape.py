import requests
import bs4
import csv


BASE_URL = 'https://www.summerhealth.com'
TOPICS_URL = BASE_URL + '/topics'


def get_qas_description(topic_url):
    topic_soup = get_soup(topic_url)
    qas = []

    for d in topic_soup.find_all('div', {'class':'faq_accordion'}):
        question = d.find('div', {"class":"faq_question"}).text[:-2]
        answer = d.find('div', {"class":"faq_answer"}).text
        qas.append([question, answer])

    description = topic_soup.find_all('div', {'class':'w-richtext'})[0].text
    
    return qas, description

def format_q(s):
    x = "are" if s.endswith('s') else "is"
    return f"What {x} " + s.replace('-', ' ') + "?"


def extract(soup):
    topics_urls = []

    for d in soup.find_all('div', {'class':'team-item care-team w-dyn-item'}):
        topics_urls.append(d.a.get('href'))

    qa_data = [["question", "answer"]]

    for topic_url in topics_urls:
        qas, description = get_qas_description(topic_url=BASE_URL + topic_url)
        qa_data.extend(qas)
        qa_data.append([format_q(topic_url[8:]), description])
    
    return qa_data

# takes a tuple of columns and rows and writes them to a csv file
def write(data, filename):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(data)

def get_soup(url):
    response = requests.get(url)
    return bs4.BeautifulSoup(response.text, features='html.parser')


def main():
    soup = get_soup(TOPICS_URL)

    qa = extract(soup)

    write(data=qa, filename='qa.csv')

    # write(data=description, filename='description.csv')



if __name__ == '__main__':
    main()