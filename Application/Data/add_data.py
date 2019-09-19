import psycopg2;
import csv;

quary_info = "\
        INSERT INTO article_info (article_id, url, category, sub_category, tags,\
        source, published, headline, classified)\
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);\
        "

quary_content = "\
        INSERT INTO article_content (article_id, content_order, content,\
        html_type, class, class_conflict)\
        VALUES (%s, %s, %s, %s, %s, %s);\
        "

def apply_data_to_database():
    try:
        conn = psycopg2.connect("dbname='postgres' user='postgres' host='db' port='5432'")
    except:
        print("ERROR: could not connect to database, in apply_data_to_database")
        return -1

    cur = conn.cursor()

    with open('./Data/article_info.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            #print(tuple(row))
            cur.execute(quary_info, tuple(row))

    with open('./Data/article_content.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            #print(tuple(row))
            cur.execute(quary_content, tuple(row))

    conn.commit()
    cur.close()
    conn.close()
    return 1

def check_if_data_is_added():
    try:
        conn = psycopg2.connect("dbname='postgres' user='postgres' host='db' port='5432'")
    except:
        print("ERROR: could not connect to database, in check_if_data_is_added")
        return -1

    cur = conn.cursor()
    cur.execute("select count(*) from article_info")

    value = cur.fetchall()
    print(value[0][0])
    if (value[0][0] != 0):
        conn.commit()
        cur.close()
        conn.close()
        return 2

    conn.commit()
    cur.close()
    conn.close()
    return 1

if (check_if_data_is_added() == 2):
    print("Data is allready added")
else:
    print("Data is not added, procceeds to add data")
    apply_data_to_database()
