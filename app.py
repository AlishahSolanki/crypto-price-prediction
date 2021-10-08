from flask import Flask
from flask_restful import Resource, Api
from Currency import Currency
from flask import request
import json


app = Flask(__name__)
api = Api(app)

class CryptoAPI(Resource):
    def get(self):
        args = request.args
        print(args["coin"])
        try:
            targets, preds = Currency().getData(args["coin"])
        except ValueError:
            print("Oops!  That was no valid number.  Try again...", ValueError)

        return {'data':json.loads(preds.to_json(orient='table').replace("\'", ''))}

api.add_resource(CryptoAPI, '/app')

if __name__ == '__main__':
    app.run(debug=True)