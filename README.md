# Network Traffic Forecasting ML App

![Network Traffic](network_traffic_image.jpg) <!-- Add an appropriate image related to network traffic or the project -->

Welcome to the Network Traffic Forecasting ML App! This project focuses on utilizing a powerful Long Short-Term Memory (LSTM) model to predict average network traffic. By leveraging machine learning techniques, we aim to provide accurate forecasts for network traffic patterns, enabling better resource allocation and planning.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#Installation)
- [Testing](#Testing)
- [Deployment with docker](#deployment)

## Introduction

Network traffic forecasting plays a crucial role in optimizing network performance, capacity planning, and resource allocation. This ML App combines the power of LSTM, a type of recurrent neural network (RNN), with the user-friendly deployment capabilities of Flask and Docker. The goal is to provide accurate forecasts for network traffic based on input CSV files.

## Features

- **LSTM Model:** Our ML App employs a sophisticated LSTM model capable of capturing temporal patterns in network traffic data.

- **CSV Input:** The app accepts CSV files as input for network traffic data.

- **Accurate Forecasts:** By utilizing historical network traffic patterns, the model aims to provide accurate forecasts that can aid in effective decision-making.

## Installation:

   1. Clone this repository to your local machine.
      ```bash
       git clone https://github.com/Valen1oussama/ML-app.git
   
   2. Prepare a virtual environment (venv) to run the Flask application. If you're unfamiliar with this, you can refer to the Flask documentation for detailed instructions:
        https://flask.palletsprojects.com/en/2.3.x/
   
   3. Once your virtual environment is set up and activated, navigate to the project directory and run the app.
      ```bash
       flask run
      
   
## Testing:
After you've successfully started the Flask application as described in the "Installation![Network-Traffic-Analysis](https://github.com/Valen1oussama/ML-app-for-network-traffic-forecasting./assets/106777178/d00e48de-1a69-4dcb-8ad9-8cfac6ab94ef)
" section, keep the Flask app running in the background.
Open a new terminal window and navigate to the project directory and run test script.
The test script will programmatically send a sample CSV file to the running Flask application and retrieve the predicted network traffic forecast.

## Deployment with docker:
 You can also run the app from a container ,try this command:
 ```bash
       docker compose up --build

Then run the test from other terminal ,Happy forecasting! If you have any questions or need further assistance, don't hesitate to ask.
   
  
   










   
