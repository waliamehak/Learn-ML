from pymongo import MongoClient


def access():
    client = MongoClient("mongodb+srv://user_1:USER_1@cluster0.0oqke.mongodb.net/<dbname>?retryWrites=true&w=majority")
    db = client.get_database('learnml_db')
    db_data = db.algorithms
    return db_data


def update(db_data, algo_name, update_data, success_message, error_message):
    try:
        db_data.update_one({'name': algo_name}, {'$set': update_data})
        update_message = success_message
    except:
        update_message = error_message

    return update_message


def find(db_data, algo_name):
    return db_data.find_one({'name': algo_name})
