import pymongo

# Create connection to MLab Database
uriLocal = "mongodb://127.0.0.1:27017"
uriLive = "mongodb://<SuperUser>:<Password>@ds123499.mlab.com:23499/draw_and_guess"
# uriLive = ""
client = pymongo.MongoClient(uriLocal)
database = client['draw-and-guess']


# Create retrieval method of JSON objects

# Create post method of JSON object

