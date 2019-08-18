import logging
import os
import string
import csv
import nltk
import glob
import ntpath
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

directory = 'ALLtextFiles'


for filename in os.listdir(directory):
     
    try:
        file=open(directory+'/'+filename,"r")
    
        text = file.read().replace('\n',' ')
        # Tokenizing the content of the whole file
        tokenizer = RegexpTokenizer(r'\w+')

        tokens = tokenizer.tokenize(text)

        # Convert to lower case 
        words = [w.lower() for w in tokens] 

        # Remove the words with numbers
        words = [word for word in words if word.isalpha()] 

        # Filter out stop words 
        stop_words = set(stopwords.words('english')) 
        words = [w for w in words if not w in stop_words]

                    
        # Stemming of the words
        stemmer = PorterStemmer()
        stemmedData = [stemmer.stem(s) for s in words] 
                    

        # Lamentization of the words
        lemmatiser = WordNetLemmatizer()
        lemmantedData = [lemmatiser.lemmatize(l) for l in stemmedData] 
        print(lemmantedData)

        file1 = open("ALLtextFilesPROCESSED/"+filename , 'w')
        for item in lemmantedData:
                file1.write(item + " ")
        #print("File number : " + str(datum['id'])+"processed.") 
    except:
        pass
