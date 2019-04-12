# Disaster Response Pipeline Project

## Table of Contents

- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Instructions](#instructions)
- [Project Organization](#project-organization)

## Description

This is a [Udacity nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) project (9. Data Engineering).

This project uses disaster data from [Figure Eight](https://www.figure-eight.com) to build a model for an API that classifies disaster messages, and deploy a Flask based web app.

## Dependencies

* Python3.6

## Installation

```python
pip install -r requirements.txt
```

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Organization

```text
[project root]
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └── process_data.py
├── models
│   └── train_classifier.py
├── README.md
└── requirements.txt
```
