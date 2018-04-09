import pymongo
import json

class DataBaseController:

    def __init__(self, islocal=False,
                 uri="mongodb://SuperUser:Password@ds123499.mlab.com:23499/draw_and_guess"
                 ):
        # Create connection string attributes
        self.islocal = islocal
        self.uri = uri
        uriLocal = "mongodb://127.0.0.1:27017"
        if islocal:
            self.client = pymongo.MongoClient(uriLocal)
            self.db = self.client['local']
            print('is Connected')
        else:
            self.client = pymongo.MongoClient(uri)
            self.db = self.client['draw_and_guess'].get_collection(name="drawings")

    def insert_drawing(self):
       picture = DGObjectModel()
       picture.parse_json_request()
       self.db.insert(picture)

       return True;

    def get_all_drawings(self):
       return self.db.find()


d = DataBaseController()

print(list(d.db.find()))
# collection = db['drawings']

        class DGObjectModel:

            def __init__(self):
                self.data = {}
                self.data['word'] = ""
                self.data['key_id'] = 12345
                self.data['country_code'] = ""
                self.data['timestamp'] = ""
                self.data['recognized'] = False
                self.data['image'] = []


            def parse_json_request(self,json_passed):
                #json passed will be the object that is passed in
                #TODO write parser for json object that is passed from front end
                new_j = json.dumps(self.data)
                return new_j
