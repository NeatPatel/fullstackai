# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import predict

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.route('/prediction', methods=["POST"])
def prediction():
    if request.method == "POST":
        #return predict.predict(request.)
        request_data = request.get_json()
        return jsonify(predict.predict(request_data['url']))
    
    return None

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(host='0.0.0.0', port=5050)
    # '0.0.0.0' makes the server accessible
    # from any IP address on your network
    # port=num is the port number 'num' to use