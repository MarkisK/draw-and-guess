import pymongo
# needs access to view object
# needs access to model object


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
            self.db = self.client['draw_and_guess']

    def insert_drawing(self):

        return

    def get_all_drawings(self):
        # TODO GET ALL DRAWINGS FROM DATABASE
        return


d = DataBaseController()

print(list(d.db['apples'].find()))
# collection = db['drawings']

