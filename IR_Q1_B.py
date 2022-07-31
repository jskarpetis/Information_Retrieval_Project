from elasticsearch import Elasticsearch
elasticsearch = Elasticsearch(host="localhost", port=9200)


# Question 1
def basic_search(field, value):
    try:
        res = elasticsearch.search(index="bx-books", query={
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
        print("Error occurred!")
        return

    if (res["hits"]["total"]["value"] == 0):
        print("No data has been returned!")
        return

    print("\nShowing the first 10 answers.\n")

    for hit in res["hits"]["hits"]:
        print("Elastics score:", hit["_score"], "--- Isbn:", hit["_source"]
              ["isbn"], "---", f"{field}:",  hit["_source"][f"{field}"])

    print("\nTotal hits -> {}\n".format(res["hits"]["total"]["value"]))


if __name__ == "__main__":
    print("Searching on index: bx-books.")
    print("You can search any field and value. Program ends with field input 'END'\n")

    while True:

        field = input("Give me a field: ")
        if (field == 'END'):
            print("Program exiting.")
            break
        value = input("Give me a value: ")
        basic_search(field, value)
