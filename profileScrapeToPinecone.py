from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import time
import numpy as np
import pandas as pd
import json
import pinecone
import pickle
import os
import spacy
import nltk
from rake_nltk import Rake

#set path to chromedriver that Selenium can access
path = "./chromedriver_mac64/chromedriver"
driver = webdriver.Chrome(path)

# Login to linkedin
def login():
    email = "your-linkedin-email"
    password = "your-linkedin-password"

    driver.get("https://www.linkedin.com/login")
    time.sleep(5)

    #login by element, clicking and filling on appropriate boxes
    eml = driver.find_element(by=By.ID, value="username")
    eml.send_keys(email)
    time.sleep(3)
    passwd = driver.find_element(by=By.ID, value="password")
    passwd.send_keys(password)
    loginbutton = driver.find_element(by=By.XPATH, value="//*[@id=\"organic-div\"]/form/div[3]/button")
    time.sleep(2)
    loginbutton.click()
    #wait for the page to load
    time.sleep(10)

#Scrape a profile for the experience and education tabs
def scrapeProfile(profile):
  currentProfile = []

  #scrape Experience page
  driver.get(profile)
  time.sleep(3)
  
  try:
    bio =  driver.find_element(by=By.ID, value="about")
    about = bio.find_element_by_xpath("..")
    currentProfile.append(about.text)
  except:
    currentProfile.append(" ")
    print(profile, " doesn't have an about section")
  

  #scrape Experience page
  driver.get(profile + "details/experience/")
  time.sleep(3)
  experience =  driver.find_element(by=By.ID, value="main")
  currentProfile.append(experience.text)
  
  #scrape Education page
  driver.get(profile + "details/education/")
  time.sleep(3)
  education =  driver.find_element(by=By.ID, value="main")
  currentProfile.append(education.text)

  return currentProfile

#format data by removing all '\n' from the scraped text and returning a string of the top 25 keywords
def extractKeywords(profileData):
  nltk.download('stopwords')
  nltk.download('punkt')
  rake_nltk_var = Rake()

  keywords = []
  for profiles in profileData:
    keywordProfile = []
    for parts in profiles:
      current = parts
      # remove dates from experience tab to avoid having too many dates in keywords
      while current.find("·") != -1:  
        pos = current.find("·")
        nextN = current.find("\n" , pos)
        prevN = current.find("\n" , max(pos - 50,0))
        current = current[:min(abs(prevN),pos - 1)]+current[max(nextN,pos + 1):]
      #get Keywords using rake-nltk
      rake_nltk_var.extract_keywords_from_text(current)
      keywordProfile.append(' '.join(rake_nltk_var.get_ranked_phrases()[:10]))
    keywords.append(keywordProfile)
  return keywords

#extend the education embedding by the education vector
def extendVec(embeddedData):
  extended = []
  for profiles in embeddedData:
    extended.append(profiles[0] + profiles[1] + profiles[2])
  return extended

#upload vectors to pinecone
def uploadVecToPinecone(vectors, profileURLS):
  cutProfileURLS = []
  listVectors = []
  #remove  'https://www.linkedin.com/in/' from profiles and use as vector titles
  for url in profileURLS:
    cutProfileURLS.append(url[28:])
  for data in vectors:
    listVectors.append(data.tolist())

  INDEX_NAME,INDEX_DIMENSION = 'profiles', (768)

  #open connection with pinecone
  pinecone.init(
  api_key="your-pinecone-api-key",
  environment='us-east1-gcp', # find in console next to api key
  )

  #find of create wanted index
  if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME,dimension=INDEX_DIMENSION, metric = 'euclidean')

  index = pinecone.Index(index_name=INDEX_NAME)
  #upload vectors to that index
  index.upsert(vectors=zip(cutProfileURLS, listVectors))

#read URLs from text file
def getURLS(file):
  file = open(file, 'r')
  return file.read().splitlines()

#write vector embeddings to file using pickle
def write_list(dataAsList, listFile):
    # store list in binary file
    with open(listFile, 'wb') as fp:
        pickle.dump(dataAsList, fp)


if __name__ == "__main__":
    print("Getting profile URLs")
    profileURLS = getURLS("profilesList.txt")
    
    profileData = []
    #login and scrape all profiles from file
    login()
    time.sleep(2)
    for profile in profileURLS:
      profileData.append(scrapeProfile(profile))
      print("Scraped profile:  " + profile)

    #close the chrome driver
    driver.quit()

    #write raw data to text file
    write_list(profileData, "rawData.txt")
    #profileData = pickle.load( open( "rawData.txt", "rb" ) )

    #use rake-nltk to format and pull keywords from data
    formattedData = extractKeywords(profileData)

    #extend experinece embedding with education embedding and bio if it exists
    extended = extendVec(formattedData)

    embeddedData = []
    #load SBERT 'all-mpnet-base-v2' model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    #only use for 300 characters of keywords
    model.max_seq_length = 300
    
    #embed each experience and education text for each profile
    for i in range(len(profileURLS)):
      profileEmbed = model.encode(extended[i])
      embeddedData.append(profileEmbed)
      print("Embedded data:  " + profileURLS[i])


    #write data to file using pickle
    write_list(embeddedData, "serializedData.txt")
    print("wrote data to file using pickle")

    #upload extended vectors to pinecone, keep first vector for query later
    uploadVecToPinecone(embeddedData[1:], profileURLS[1:])
    print("Uploaded embeddings to pinecone except first profile")

    