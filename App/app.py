#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:18:18 2020

@author: celinaagostinho
"""

# import flask magic
# import render_template : allows you to have html code in your app
# import request
from flask import Flask, render_template, request

# from filename import function we defined
from translate_es import translate

# Flask logic section
# creating an object that is a flask app
# defines the app
app = Flask(__name__)

# we have to create route
# A route is the door (or a possible door) that our app can expect to be opened

# decorator
# we use decorators to create routes
# calls a python inbuilt function wihtout having to import anything
# defines the route - e.g. @app.route('/celina/')
# create an empty route
@app.route('/', methods = ['POST', 'GET'])

# for each route define one python function to be executed
def index():
    # python logic that'll be run when the route is called
    # return "Out of Scope"
    if request.method == 'POST':
        try:
        # POST request : run model button on html page
        # Do stuff : define the parameters
            sentence = str(request.form['sentence'])
        
            prediction = translate(sentence)
            
            #print('The predicted translation is', prediction)
            
            # if we want to return the result to our html, we need to pass it as a 
            # variable to the html
            # [prediction[:-6]] because it expects a list, and we don't want to print <end>
            return render_template('main.html', predictions = [prediction[:-6]])
        
        except:
            return render_template('main.html')
            
    else:
    # renders the html
        return render_template('main.html')

# running our flask app defined above
# normally debug should be True in development
if __name__ == "__main__":
    app.run(debug = True)

    

