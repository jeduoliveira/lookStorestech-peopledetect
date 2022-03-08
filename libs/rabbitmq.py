# -*- coding: utf-8 -*-
# pylint: disable=C0111,C0103,R0205

import pika
import logging
import json
from pika.exchange_type import ExchangeType
#from botocore.exceptions import ClientError

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RabbitMQ:
    
    EXCHANGE = None
    def __init__(self, host, port, username, password):
        self.EXCHANGE = config['rabbitmq']['exchange']
        self._connection = None
        self._channel = None
        self._stopping = False
        self._host = host
        self._port = port
        self._username = username
        self._password = password

    def connect(self):
        LOGGER.info('Connecting to %s', self._host)
        self._connection =  pika.BlockingConnection(
            pika.ConnectionParameters(
                            host=self._host,
                            port=self._port,
                            credentials=pika.PlainCredentials(self._username, self._password),
                            heartbeat=600,
                            blocked_connection_timeout=300)
        )
        
    def open_channel(self):
        LOGGER.info('Creating a new channel')
        self._channel = self._connection.channel()
        
    def setup_queue(self, queue_name):
        LOGGER.info('Declaring queue %s', queue_name)
        self._channel.queue_declare(
            queue=queue_name, auto_delete=True)

    def setup_exchange(self):
        LOGGER.info('Declaring exchange %s', self.EXCHANGE)
        self._channel.exchange_declare(
            exchange=self.EXCHANGE, 
            exchange_type=ExchangeType.direct,
            passive=False,
            durable=True,
            
            auto_delete=False)

    def setup_binding(self, routingKey, queue_name):
        self._channel.queue_bind(exchange=self.EXCHANGE,
                   queue=queue_name,
                   routing_key=routingKey)
        self._channel.basic_qos(prefetch_count=1)

    def stop(self):
        LOGGER.info('Stopping')
        self._stopping = True
        self.close_channel()
        self.close_connection()

    def close_channel(self):
        if self._channel is not None:
            LOGGER.info('Closing the channel')
            self._channel.close()

    def close_connection(self):
       
        if self._connection is not None:
            LOGGER.info('Closing connection')
            self._connection.close()

    def publish_message(self, routingKey, message): 
        self.connect()
        self.open_channel()

        try:
            self.setup_exchange()
            self.setup_queue(routingKey)
            self.setup_binding(routingKey, routingKey)

            self._channel.basic_publish(
                exchange=self.EXCHANGE, 
                routing_key=routingKey,
                body=json.dumps(message, ensure_ascii=False),
                properties=pika.BasicProperties(content_type='application/json'))
        except: 
            print("Send message error")
        finally:
            self.stop()
                    

def main():
    mq = RabbitMQ('amqp://localhost:5672/%2F?connection_attempts=3&heartbeat=3600')

if __name__ == "__main__":
    main()