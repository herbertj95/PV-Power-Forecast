"""
How to publish?

The only function needed to publish to a topic is the "publish" function. In this function, it is going to be sent a message
to a broker in order to be forward to the subscribers.

The function return 0 if the message was sent
Else returns a negative number
"""


"""
Errors:
1-> To much arguments
2-> Couldn't split arguments correctly
3-> The input in the main funciton is not a string
4-> Couldn't read the input
5-> To few arguments in the topic
6-> Error! Could not send the message to the cloud
7-> Error! Could not send the message to the topic
8-> Topic is not in the correct format ("ssop/#")
9-> Could not convert dict into string for some reason
"""

from paho.mqtt import client as mqtt_client
import pandas as pd
import json

#Messges must be shoreter than 10000 carachters
MAX_LEN_MESSAGE = 10000
#The topic given cannot be longer than x arguments
MAX_N_OF_ARGUMENTS = 10

#Password and username to be implemanted later 
USERNAME = 'selssopcloud2'
PASSWORD = 'jp4dg'

CLOUD_TOPIC = 'ssop/SSOPCloud'

#Online broker information
broker = 'mqtt-broker.smartenergylab.pt'
port = 1883
    
#First, connect to the broker
def connect_mqtt(client_id):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

#Send the message
def sendMessage(client,message,topic):

    try:
        data = str( message)

    except:
        print("Couldn't convert Dict to String")
        return -9

    status = client.publish(topic ,data)[0]
    
    if status != 0:
        print(f"Could not send: \n\n`{data}`\n to the cloud")
        return -6
    else:
        print(f"Message '{data}' sent!")
        return 0


#Main function used to publish topics
def publish(message, client_id, topic = CLOUD_TOPIC , username=USERNAME, password=PASSWORD):
    
    if len(message) > MAX_LEN_MESSAGE:
        return -2

    try:
        splitTopic = topic.split('/', MAX_N_OF_ARGUMENTS)
    except:
        return -3
    
    if splitTopic[0] != 'ssop'  or (not isinstance(client_id, str)):
        print("The topic must be in the ssop/# format!")
        return -8
    if not isinstance(client_id, str):
        print("The clientID must be a string")
        return -8

    if type(message) is not dict:
        print("Input is not in the correct format: Must be a dictionary")
        return -3
    
    client = connect_mqtt(client_id)
    return sendMessage(client,message,topic)


#Function to create the format of the data to send
def convertToJson(dict):

    keysList = list(dict.keys())
    valuesList = list(dict.values())

    dict = {
        'prediction' : {
            'keys' : keysList,
            'values' : valuesList,
        } 
    }

    return dict


'Reading the data to send'
data2 = pd.read_csv('./Forecast/Predictions_XGBOOST.csv')
data2.astype(str)
data2 = dict(zip(data2.Date, data2.Prediction.values))
data2 = convertToJson(data2)

data = {
    "credentials" : {
        "ID" : "pedro",
        "password" : "passss"
    },
    "dataType" : "AlgorithmInformation",
    "data" : data2
    }

'Sending the data as a message'
publish(data, 'Herbert')
