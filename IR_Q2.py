
import re
from elasticsearch import Elasticsearch
import pandas as pd
from pyparsing import col
elasticsearch = Elasticsearch(hosts="localhost", port=9200)


# Question 2
def new_metric(elastics_score, average_rating, user_rating):

    if user_rating is not None and average_rating is not None:
        final_score = elastics_score + 2*user_rating + average_rating

    elif user_rating is None and average_rating is None:
        final_score = elastics_score

    elif user_rating is not None and average_rating is None:
        final_score = elastics_score + 2*user_rating

    elif user_rating is None and average_rating is not None:
        final_score = elastics_score + average_rating

    return final_score


def basic_search(field, value):
    try:
        res = elasticsearch.search(index="bx-books", from_=0, size=10000, query={
            "match": {
                f"{field}": f"{value}"
            }},
            sort=[
                {
                    "_score": {
                        "order": "desc"
                    }
                }
        ]
        )
    except:
        print("There has been an error in retrieving the data!")
        return

    if (res["hits"]["total"]["value"] == 0):
        print("No data has been returned!")
        return

    return res


# Finds the rating that a specific uid has given to a book of specific isbn returning it as an integer, if it doesn't find any rating then it returns None
def user_rating_search(uid, isbn):
    try:
        res = elasticsearch.search(index="bx-book-ratings", query={
            "bool": {
                "must": [
                    {"match": {
                        "isbn": f"{isbn}"
                    }
                    },
                    {
                        "match": {
                            "uid": f"{uid}"
                        }
                    }

                ]
            }
        })
    except:
        print("There has been an error in retrieving the data!")

    if (res["hits"]["total"]["value"] == 0):
        return
    else:
        hit_rating = res["hits"]["hits"][0]["_source"]["rating"]
        return float(hit_rating)


# Calculates the average rating given to a book, note that we don't use the specified users rating in this calculation
def calculate_all_user_ratings(uid, isbn):
    total_rating = 0
    counter = 0
    try:
        res = elasticsearch.search(index="bx-book-ratings", from_=0, size=10000, query={
            "bool": {
                "must": [
                    {"match": {"isbn": f"{isbn}"}}
                ],
                "must_not": [
                    {"match": {
                        "uid": f"{uid}"
                    }}
                ]
            }
        })
    except:
        print("There has been an error in retrieving the data!")
        return

    if res["hits"]["total"]["value"] == 0:
        print("There are no ratings for this book")
        return

    for hit in res["hits"]["hits"]:
        counter += 1
        total_rating += float(hit["_source"]["rating"])

    average_rating = total_rating / counter
    return float(average_rating)


if __name__ == "__main__":
    resulting_df = pd.DataFrame(columns=[
                                'Book_Isbn', 'Elastics_Score', 'Specified_Users_Rating', 'Average_Rating', 'Final_Score'])

    print("Searching on index: bx-books.")
    print("You can search any field and value. Program ends with user_id input 'END'\n")

    while True:
        # uid: 253580
        user_id = input("Give me a user_id(uid): ")
        if (user_id == 'END'):
            print("Program exiting.")
            break
        field = input("Give me a field: ")
        # book_title
        value = input("Give me a value: ")
        # Fatal Voyage
        results = basic_search(field, value)

        if (results["hits"]["total"]["value"] == 0):
            print("No data has been returned!")
        else:
            print("Total hits -> {}".format(results["hits"]["total"]["value"]))
            print("For user -> {}".format(user_id))

            for result in results["hits"]["hits"]:

                elastics_score = result["_score"]

                specified_users_score = user_rating_search(
                    user_id, result["_source"]["isbn"])

                average_rating = calculate_all_user_ratings(
                    user_id, result["_source"]["isbn"])

                final_score = new_metric(
                    elastics_score, average_rating, specified_users_score)

                book_isbn = result["_source"]["isbn"]

                resulting_df.loc[len(resulting_df.index)] = [
                    book_isbn, elastics_score, specified_users_score, average_rating, final_score]
            resulting_df = resulting_df.sort_values(
                'Final_Score', ascending=False)
            resulting_df = resulting_df[resulting_df.columns[::-1]]
            print(resulting_df.head(15))
            # print("\nBook_isbn -> {}\t\t Elastics_Score -> {:.2f}\t\t Specified_Users_Rating -> {}\t\t Average_Rating -> {}\t\t Final_Score -> {:.2f}".format(
            #     result["_source"]["isbn"], elastics_score, specified_users_score, average_rating, final_score))

            # Gina Bari Kolata
