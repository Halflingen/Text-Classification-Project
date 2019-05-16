import psycopg2;


def get_dataset():
    quary =\
    """
    select *
    from article_content
    where class is not null
    """

    try:
        conn = psycopg2.connect("dbname='master_arx' user='aanund'")
    except:
        print("ERROR: could not connect to database")
        return -1

    cur = conn.cursor()
    cur.execute(quary)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def store_result(data):
    quary =\
    """
    INSERT INTO predictionScores(model, accuracy, recall, precision, f1, mcc,
    ignore_f1, goal_assist_f1, transfer_f1, quote_f1, irrelevant_f1,
    training_time, computation_time)
    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
    """

    try:
        conn = psycopg2.connect("dbname='master_arx' user='aanund'")
    except:
        print("ERROR: could not connect to database")
        return -1

    cur = conn.cursor()
    cur.execute(quary, data)
    id_ = cur.fetchone()

    conn.commit()
    cur.close()
    conn.close()

    return id_


def store_parameters(quary, data):
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


def check_if_exists(quary, data):
    try:
        conn = psycopg2.connect("dbname='master_arx' user='aanund'")
    except:
        print("ERROR: could not connect to database")
        return -1

    cur = conn.cursor()
    cur.execute(quary, data)

    value = cur.fetchall()[0][0]
    conn.commit()
    cur.close()
    conn.close()
    return value
