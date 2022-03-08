#!/usr/bin/env python
#
# Example consumer of queue messages
#
# pip3 install -r requirements.txt
#
import argparse
import sys
import os
import pika
import signal
import cv2
import pymongo
import time
import json
import boto3
import numpy as np
import configparser

import base64
from PIL import Image
from io import BytesIO


config = configparser.ConfigParser()
config.read('config.ini')

rekognitionClient=boto3.client('rekognition')

def queue_callback(channel, method, properties, body):
  if len(method.exchange):
    #with open('./data/facedetection.txt', 'a') as f:
    #  f.write(body.decode('UTF-8') + '\n')

    #print("from exchange '{}': {}".format(method.exchange,body.decode('UTF-8')))

    dJson = json.loads(body.decode('UTF-8'))
    data = dJson['lookstorestech.facedetection']['data']

    mongoClient = pymongo.MongoClient("mongodb+srv://lookstoretech:F7LlsMjEOofwMkQc@lookstoretech-cluster.q1oza.mongodb.net/lookstoretech?retryWrites=true&w=majority")
    db = mongoClient.lookstoretech
    
    #with mongoClient:
      #im = Image.open(BytesIO(base64.b64decode(data['image'])))
      #im.save('image1.png', 'PNG')

      #load_bytes = bytes(data['image'],'UTF-8')
      #loaded_np = np.load(load_bytes, allow_pickle=True)
      #image = cv2.imdecode(loaded_np, cv2.IMREAD_COLOR) 
      #cv2.imshow("Output Video", image)
     
      #response = rekognitionClient.detect_faces(
      #    Image={
      #        'Bytes': image
      #    }, 
      #    Attributes=[
      #        'ALL'
      #    ]
      #)
#
      #for faceDetail in response['FaceDetails']:
      #    for emotion in faceDetail['Emotions']:
      #        print(str(emotion['Type']))
      #        objectJson = {     
      #            'store': data['store_id'] ,
      #            'name': data['name'],
      #            'age_min': str(faceDetail['AgeRange']['Low']),
      #            'age_max': str(faceDetail['AgeRange']['High']),
      #            'gender': str(faceDetail.get('Gender', {}).get('Value', None) ) ,
      #            'emotion': str(emotion['Type']),
      #            'emotion_confidence': str(emotion['Confidence']),
      #            'date': data['date']
      #        }
      #        #db.peopledetect.insert_one(objectJson)
  
    

  #else:
  #  print("from queue {}: {}".format(method.routing_key,body.decode('UTF-8')))

def signal_handler(signal,frame):
  print("\nCTRL-C handler, cleaning up rabbitmq connection and quitting")
  connection.close()
  sys.exit(0)

example_usage = '''====EXAMPLE USAGE=====
Connect to remote rabbitmq host
--user=guest --password=guest --host=192.168.1.200
Specify exchange and queue name
--exchange=myexchange --queue=myqueue
'''

# connect to RabbitMQ
credentials = pika.PlainCredentials(config['rabbitmq']['username'], config['rabbitmq']['password'] )
connection = pika.BlockingConnection(pika.ConnectionParameters(config['rabbitmq']['host'], config['rabbitmq']['port'], '/', credentials ))
channel = connection.channel()

channel.basic_consume(queue="lookstorestech.facedetection", on_message_callback=queue_callback, auto_ack=True)

# capture CTRL-C
signal.signal(signal.SIGINT, signal_handler)

print("Waiting for messages, CTRL-C to quit...")
print("")
channel.start_consuming()