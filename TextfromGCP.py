from google.cloud import storage
import logging
import os
import json
import pandas as pd


def list_blobs(bucket_name):
	# Reading the CSV file
	data = pd.read_csv("Other.csv" ,usecols = ['id','type'])
	storage_client = storage.Client()

	# Reading the bucket using the bucket Name
	bucket = storage_client.get_bucket(bucket_name)

	# Iterating through each row of the data
	for row, datum in data.iterrows():

		# Reading a particular folder using the ID of the folder
		blobs = bucket.list_blobs(prefix = f"Valkyrie/"+datum['id']+".pdf")

		# Creating a file using the ID of the Folder
		file = open("Other/"+datum['id']+".txt", "w")

		# Printing to know the status
		print("File : "+datum['id'])

	for blob in blobs:
		# Download the JSON blob data as string and then load it as JSON
		json_data = json.loads(blob.download_as_string())

		try:
		# Extracting the text from the JSON object
		text = json_data['responses'][0]['fullTextAnnotation']['text']

		# Writing the text into the file
		file.write(text)
		# Spacing for clarification
		file.write("**************************** \n")
		except:
		pass

list_blobs("valkyrie-ocr-dev")
