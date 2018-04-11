import pymongo
from parser import DGObjectModel


class DataBaseController:

    def __init__(self, islocal=False,
                 uri="mongodb://SuperUser:Password@ds123499.mlab.com:23499/draw_and_guess"
                 ):
        # Create connection string attributes
        self.islocal = islocal
        self.uri = uri
        uri_local = "mongodb://127.0.0.1:27017"
        if islocal:
            self.client = pymongo.MongoClient(uri_local)
            self.db = self.client['local']
            print('is Connected')
        else:
            self.client = pymongo.MongoClient(uri)
            self.db = self.client['draw_and_guess'].get_collection(name="drawings")

    def insert_drawing(self):
        picture = DGObjectModel()
        picture.parse_json_request()
        self.db.insert(picture)
        return True

    def get_all_drawings(self):
        return self.db.find()
