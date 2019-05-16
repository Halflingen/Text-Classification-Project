import psycopg2;

def insert_into_database(quary, data):
    try:
        conn = psycopg2.connect("dbname='master_arx' user='aanund'")
    except:
        print("ERROR: could not connect to database")
        return -1

    cur = conn.cursor()
    cur.execute(quary, data)

    conn.commit()
    cur.close()
    conn.close()
    return 1

def check_if_exists(id_):
    try:
        conn = psycopg2.connect("dbname='master_arx' user='aanund'")
    except:
        print("ERROR: could not connect to database")
        return -1

    cur = conn.cursor()
    cur.execute("\
            select exists(select from new_article_info where article_id = %s);\
        ",
        (id_,))

    value = cur.fetchall()[0][0]
    conn.commit()
    cur.close()
    conn.close()
    return value
