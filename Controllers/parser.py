# import json
# Parser need to create information for neural
from pprint import pprint


class DGObjectModel:

    def __init__(self):
        self.data = {'word': 'test', 'key_id': 1234, 'country_code': 'us',
                     'timestamp': "", 'recognized': False,
                     'image': []}

    def parse_json_request(self, json_passed):
        # TODO write parser for json string object that is passed from front end

        # print(type(json_passed))
        # pprint(json_passed)

        tup = json_passed.get('shapes')

        pprint(list(json_passed.get('shapes')))

        self.data['word']
        self.data['key_id']
        self.data['country_code']
        self.data['timestamp']
        self.data['recognized']
        self.data['image'] = tup

        return self.data




