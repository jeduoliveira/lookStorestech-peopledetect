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
import pymongo
import time
import json

import configparser
config = configparser.ConfigParser()
config.read('config.ini')


def queue_callback(channel, method, properties, body):
  if len(method.exchange):
    client = pymongo.MongoClient("mongodb+srv://lookstoretech:F7LlsMjEOofwMkQc@lookstoretech-cluster.q1oza.mongodb.net/lookstoretech?retryWrites=true&w=majority")
    db = client.lookstoretech


    with open('./data/peoplecount.txt', 'a') as f:
      f.write(body.decode('UTF-8') + '\n')

    print("from exchange '{}': {}".format(method.exchange,body.decode('UTF-8')))

    peopleJson = json.loads(body.decode('UTF-8'))
    data = peopleJson['lookstorestech.peoplecount']['data']

    with client:
      db.peopledetect.insert_one(data)

    time.sleep(10)

  else:
    print("from queue {}: {}".format(method.routing_key,body.decode('UTF-8')))

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

channel.basic_consume(queue="lookstorestech.peoplecount", on_message_callback=queue_callback, auto_ack=True)

# capture CTRL-C
signal.signal(signal.SIGINT, signal_handler)

print("Waiting for messages, CTRL-C to quit...")
print("")
channel.start_consuming()