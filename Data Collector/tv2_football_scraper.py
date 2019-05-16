import requests
import psycopg2;
from bs4 import BeautifulSoup
from urllib.parse import urlparse
#from fake_useragent import UserAgent
import database_api

def handle_url(url):
    tags = ['sport', 'nyheter', 'rampelys', 'a']
    obj = urlparse(url)
    path_split = obj.path.split('/')

    if path_split[1] not in tags:
        return 0

    return 1

def get_unique_id(url):
    obj = urlparse(url)
    path_split = obj.path.split('/')
    return path_split[-1]


def get_valid_hrefs():
    hrefs = []
    file_ = open("./tv2.txt", "r")
    page_content = file_.read()
    soup = BeautifulSoup(page_content, 'html.parser')
    article_content = soup.find('div', {'id':'archiceArticles'})
    list_of_articles = article_content.find_all('a')

    for y in list_of_articles:
        if y.text:
            ret = handle_url(y['href'])
            if (ret == 1):
                href = y['href']
                hrefs.append(href)
    return hrefs


def get_content(soup, id_):
    quary = "\
        INSERT INTO new_article_content (article_id, content_order, content,\
        html_type, class, class_conflict)\
        VALUES (%s, %s, %s, %s, %s, %s);\
        "

    #this is not html header, but header in the body
    j = 0
    header = soup.find('header', {'class':'article-head '})
    headline = header.find('h1')
    data = (id_, j, headline.text, headline.name, None, None)
    database_api.insert_into_database(quary, data)
    j += 1

    description = header.find('div', {'itemprop':'description'})
    data = (id_, j, description.text, 'h2', None, None)
    database_api.insert_into_database(quary, data)
    j += 1

    article_content = soup.find_all('div', {'class':'bodytext '})
    for x in article_content:
        while True:
            if x.div is None: break
            x.div.decompose()
        while True:
            if x.aside is None: break
            x.aside.decompose()
        while True:
            if x.section is None: break
            x.section.decompose()

        p_in_article = x.find_all(['p', 'h1', 'h2'])
        for i in p_in_article:
            if len(i.text) < 2: continue
            data = (id_, j, i.text, i.name, None, None)
            database_api.insert_into_database(quary, data)
            j += 1

    return 1

def get_timestamp(soup):
    post_time = soup.find('section', {'class':'dateline-container'})
    puplished = post_time.find('time', {'itemprop':'datePublished'})
    time_format = "publisert 01.08.1989 12:00".split(" ")
    if puplished != None:
        time_format = puplished['title'].split(" ")

    date = time_format[1].replace(".", "-").split("-")
    time = time_format[2].split(":")

    year = int(date[2])
    month = int(date[1])
    day = int(date[0])
    hour = int(time[0])
    minutes = int(time[1])
    sec = int('00')
    return psycopg2.Timestamp(year, month, day, hour, minutes, sec)


def get_info(soup, id_, url):
    header = soup.find('head')

    headline = header.find('title').text

    tags = header.find('meta', {'property':'keywords'})
    tags_arr = [word.replace("-", " ") for word in tags['content'].split(',')]

    cateogry = 'Sport'

    sub_category = 'Fotball'

    timestamp = get_timestamp(soup)

    data = (id_, url, cateogry, sub_category, tags_arr, 'tv2', timestamp, headline, False)
    quary = "\
        INSERT INTO new_article_info (article_id, url, category, sub_category, tags,\
        source, published, headline, classified)\
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);\
        "

    database_api.insert_into_database(quary, data)
    return 1

def get_articles(href):
    id_ = get_unique_id(href)
    print(href, id_)
    if database_api.check_if_exists(id_): return
    #ua = UserAgent()
    #header = {'User-Agent':str(ua.chrome)}
    get_page = requests.get(url = href)
    soup = BeautifulSoup(get_page.text, 'html.parser')

    article_info = get_info(soup, id_, href)
    if article_info == 2: return
    article = get_content(soup, id_)

def main():
    hrefs = get_valid_hrefs()
    #get_articles(href)



if __name__ == '__main__':
    main()
