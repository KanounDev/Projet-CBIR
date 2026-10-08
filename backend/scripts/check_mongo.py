import time
from pymongo import MongoClient

db = MongoClient("mongodb://localhost:27017")["cbir"]
names = db.list_collection_names()
print("collections:", names)

# take the collection with the most documents
col = db[max(names, key=lambda n: db[n].estimated_document_count())]
print("using:", col.name, "->", col.estimated_document_count(), "documents")

t = time.time()
n = col.count_documents({"keywords": {"$all": ["sky", "sunset"]}})
print(n, "images with sky AND sunset in", round(time.time() - t, 3), "s")

t = time.time()
n = col.count_documents({"keywords": {"$in": ["sky", "sunset"]}})
print(n, "images with sky OR sunset in", round(time.time() - t, 3), "s")

print("indexes:", list(col.index_information()))