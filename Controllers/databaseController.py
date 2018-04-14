import pymongo
from parser import DGObjectModel
import json


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

    def insert_drawing(self, jsontest={}):
        picture = DGObjectModel()

        self.db.insert(picture.parse_json_request(jsontest))
        return True

    def get_all_drawings(self):
        return self.db.find()


test = json.loads('{"position":{"x":0,"y":0},"scale":1,"shapes":[{"className":"LinePath","data":{"order":3,"tailSize":3,"smooth":true,"pointCoordinatePairs":[[222,219.21875]],"smoothedPointCoordinatePairs":[[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875],[222,219.21875]],"pointSize":3,"pointColor":"hsla(0, 0%, 0%, 1)"},"id":"3b273a87-f170-b388-e49b-28d872051697"}],"imageSize":{"width":"infinite","height":"infinite"}}')

d = DataBaseController()

print(d.insert_drawing(test))