
# coding: utf-8

# In[ ]:

import numpy as np
from flask import Flask
from flask import request, abort
from flask import send_file
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import csv
import json
import os.path
import uuid
import mimetypes
from functools import wraps


# In[ ]:

database = None
model = None
 


# In[ ]:

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=None)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=None)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###
    
#     print(pos_dist, neg_dist, basic_loss)
    return loss

def load_model():
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    print("Loading weights...")
    load_weights_from_FaceNet(FRmodel)
    print("Weights loaded")
    return FRmodel

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 999.9
    min_user_id = None
    # Loop over the database dictionary's names and encodings.
#     for (name, db_enc) in database.items():
    for(user_id, data) in database.items():
        
        db_enc = np.reshape(data['encoding'], (1,-1))
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)
        print("Checking for " + database[user_id]['name'] + ", dist='" + str(dist) +"'")

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            min_user_id = user_id

    ### END CODE HERE ###
    
    if min_dist > 0.5:
        print("Not in the database.")
        min_user_id = None
    else:
        print ("it's " + database[min_user_id]['name'] + ", the distance is " + str(min_dist))
        
    return min_dist, min_user_id


def save_file(name, data):
    f = open(name, "wb", 0)
    f.write(data)
    f.flush()
    
def save_as_json_file(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)
    
def load_from_json_file(filename):
    if not os.path.isfile(filename):
         save_as_json_file(filename, {})
            
    with open(filename, 'r') as f:
        data = json.load(f)
        return data


# In[ ]:


app = Flask(__name__)

def require_api_token(func):
    @wraps(func)
    def check_token(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header is not None:
                token_parts = auth_header.split(' ')
                if len(token_parts) == 2 and token_parts[0] == 'Bearer':
                    token = token_parts[1]
        api_tokens = config['api_tokens'] 
        if token not in api_tokens:
            print("Invalid token='" + str(token) + "'")
            abort(401)
        return func(*args, **kwargs)

    return check_token


@app.route("/user", methods = ['GET','PUT'], defaults={'id': None})
@app.route("/user/<id>", methods = ['DELETE'])
@require_api_token
def handle_user(id):
    
    if request.method == 'GET':
        database = load_from_json_file('data.json')
        result = []
        for (user_id, user_data) in database.items():
            result_user_data = {}
            result_user_data['id'] = user_data['id']
            result_user_data['name'] = user_data['name']
            result_user_data['image_url'] = request.url_root + user_data['image']
            result.append(result_user_data)
        return str(result)
    elif request.method == 'PUT':
        database = load_from_json_file('data.json')
        f = request.files['image']

        user_data = {}
        user_data['id'] = str(uuid.uuid4())   
        user_data['image'] = config['image_path'] + str(user_data['id']) + ".png" 
        user_data['name'] = request.form['name']
        f.save(user_data['image'])
        user_data['encoding'] = img_to_encoding(user_data['image'], model)[0].tolist()

        database[user_data['id']] = user_data
        save_as_json_file('data.json', database)
        print("Created: '" + str(user_data['id']) + "'")
        return str({'user_id':user_data['id']} )
    
    elif request.method == 'DELETE':
        database = load_from_json_file('data.json')
        if id in database:
            user_data = database[id]
            if user_data is not None:
                if os.path.exists(user_data['image']):
                    os.remove(user_data['image'])
                del database[id]
                save_as_json_file('data.json', database)
                print("User '" + id + "' deleted.")
        return ""

    
@app.route("/identify", methods = ['POST'])
@require_api_token
def recognize_face():
    database = load_from_json_file('data.json')
    f = request.files['image']
    temp_fn = "tmp/" + str(uuid.uuid4())
    f.save(temp_fn)
    min_dist, user_id = who_is_it(temp_fn, database, model)
    os.remove(temp_fn)
    result = {}
    if user_id is None:
        return str({'status': 'not found'})  + "\n"
    return str({'status': 'success', 'user': {'user_id':user_id, 'name':database[user_id]['name']}, 'dist':min_dist}) + "\n"


@app.route("/images/<image_name>", methods = ['GET'])
@require_api_token
def image_download(image_name):
    fn = config['image_path'] + image_name
    mime_type = mimetypes.guess_type(fn)[0]
    if not os.path.exists(fn):
        abort(404)
    return send_file(fn, mimetype=mime_type)


config = load_from_json_file('config.json')
print("Start server on '" + config['host'] + ":" + str(config['port']) + "'") 
if __name__ == "__main__":
    if model is None:
        model = load_model()
    app.run(host=config['host'], port=config['port'])


# In[ ]:



