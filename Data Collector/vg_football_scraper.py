import requests
import psycopg2;
from bs4 import BeautifulSoup
from urllib.parse import urlparse
#from fake_useragent import UserAgent
import database_api

def handle_url(url):
    tags = ['sport', 'nyheter', 'rampelys']
    obj = urlparse(url)
    path_split = obj.path.split('/')

    if path_split[1] not in tags:
        return 0

    return 1

def get_unique_id(url):
    obj = urlparse(url)
    path_split = obj.path.split('/')
    for i in range(len(path_split)):
        if path_split[i] == 'i':
            return path_split[i+1]

    return -1

def get_valid_hrefs():
    hrefs = []
    file_ = open("./vg.txt", "r")
    page_content = file_.read()
    soup = BeautifulSoup(page_content, 'html.parser')
    article_content = soup.find('div', {'class':'_2PVad'})
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
    article_content = soup.find(class_="_1SeBT CZnSM")
    while True:
        if article_content.div is None: break
        article_content.div.decompose()
    while True:
        if article_content.aside is None: break
        article_content.aside.decompose()
    while True:
        if article_content.section is None: break
        article_content.section.decompose()

    p_in_article = article_content.find_all(['p', 'h1', 'h2'])

    j = 0
    for i in p_in_article:
        if len(i.text) < 2: continue
        data = (id_, j, i.text, i.name, None, None)
        database_api.insert_into_database(quary, data)
        j += 1

    return 1

def get_timestamp(soup):
    post_time = soup.find_all('time')
    time_format = post_time[0].text.replace(".", "-").split(" ")

    date = time_format[0].split("-")
    time = time_format[1].split(":")

    year = int("20" + date[2])
    month = int(date[1])
    day = int(date[0])
    hour = int(time[0])
    minutes = int(time[1])
    sec = int('00')
    return psycopg2.Timestamp(year, month, day, hour, minutes, sec)

def subscription(header, id_):
    quary = "\
        INSERT INTO new_article_info (article_id, url, category, sub_category, tags,\
        source, published, headline, classified)\
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);\
        "
    check_if_pluss = header.find('meta', {'name':'lp:type'})
    if check_if_pluss is not None:
        if check_if_pluss['content'] == 'article_subscription':
            data = (id_, "na", None, None, None, None, None, "na", False)
            database_api.insert_into_database(quary, data)
            return True
    return False

def get_info(soup, id_, url):
    header = soup.find('head')
    if subscription(header, id_): return 2

    headline = header.find('title')
    headline = headline.text[:-5]

    #tags = header.find('meta', {'name':'keywords'})
    #tags_arr = [word.lstrip().lower() for word in tags['content'].split(',')]
    tags_arr = []

    cateogry = 'Sport'

    sub_category = 'Fotball'

    timestamp = get_timestamp(soup)

    data = (id_, url, cateogry, sub_category, tags_arr, 'vg', timestamp, headline, False)
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
