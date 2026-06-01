# Forex prediction for EUR/USD

![python](https://img.shields.io/badge/python-programming-blue?style=flat-square&logo=python&logoColor=white)
![tensorflow](https://img.shields.io/badge/tensorflow-machine_learning-orange?style=flat-square&logo=tensorflow&logoColor=white)
![keras](https://img.shields.io/badge/keras-deep_learning-red?style=flat-square&logo=keras&logoColor=white)
![scikit learn](https://img.shields.io/badge/scikit_learn-data_preprocessing-orange?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data_manipulation-purple?style=flat-square&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-numerical_computing-blue?style=flat-square&logo=numpy&logoColor=white)
![apache kafka](https://img.shields.io/badge/apache_kafka-live_streaming-black?style=flat-square&logo=apachekafka&logoColor=white)
![docker](https://img.shields.io/badge/docker-container_deployment-blue?style=flat-square&logo=docker&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-market_data_api-brightgreen?style=flat-square&logo=yahoo&logoColor=white)

This project implements a real time price forecasting pipeline for the eur/usd currency pair. It combines a hybrid deep learning model with a live streaming data infrastructure.

You can read the full project report here: [project report](rapport_pfe.pdf).

## project overview
The system is built around a hybrid machine learning model that combines a multi head attention mechanism with a bidirectional gated recurrent unit. This model predicts the price variation of the next minute based on the prices of the previous sixty minutes. The live data flows continuously from yahoo finance through an apache kafka pipeline.

## system pipeline
- **Live producer:** a python script queries the yahoo finance api every sixty seconds for the latest closing rate and sends it as a message to a kafka topic.
- **Live consumer:** a prediction script consumes the data from the kafka topic, calculates the variations, normalizes the inputs, and performs real time inference using the trained model.
- **Infrastructure:** the apache kafka service is deployed in kraft mode using docker containers.

## model performance
The model was trained on five years of historical data and achieved the following results during evaluation:
- Mean absolute error: 1.18 pips
- Root mean square error: 1.59 pips
- Mean absolute percentage error: 0.0102%
- Coefficient of determination: 0.9914
- Average prediction bias: -0.71 pips

## project files
- `train_model.py`: script used to train the hybrid prediction model.
- `evaluate_model.py`: script to evaluate the performance of the model on the test dataset.
- `live_producer.py`: script that fetches live prices from yahoo finance and publishes them to the kafka topic.
- `spark_predict.py`: script that consumes the live kafka feed and performs real time predictions.
- `docker-compose.yml`: configuration to start the apache kafka service using docker.
