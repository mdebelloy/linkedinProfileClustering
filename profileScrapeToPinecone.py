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

#set path to chromedriver that Selenium can access
path = "./chromedriver_mac64/chromedriver"
driver = webdriver.Chrome(path)

# Login to linkedin
def login():
    email = "your-linkedin-email"
    password = "your-linkedin-password"

    driver.get("https://www.linkedin.com/login")
    time.sleep(1)

    #login by element, clicking and filling on appropriate boxes
    eml = driver.find_element(by=By.ID, value="username")
    eml.send_keys(email)
    passwd = driver.find_element(by=By.ID, value="password")
    passwd.send_keys(password)
    loginbutton = driver.find_element(by=By.XPATH, value="//*[@id=\"organic-div\"]/form/div[3]/button")
    loginbutton.click()
    #wait for the page to load
    time.sleep(3)

#Scrape a profile for the experience and education tabs
def scrapeProfile(profile):
  currentProfile = []

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

#format data by removing all '\n' from the scraped text
def formatData(profileData):
  formatted = []
  for profiles in profileData:
    formatedProfile = []
    for parts in profiles:
      formatedProfile.append(parts.replace("\n", " "))
    formatted.append(formatedProfile)
  return formatted

#extend the education embedding by the education vector
def extendVec(embeddedData):
  extended = []
  for profiles in embeddedData:
    extended.append(np.concatenate((profiles[0],profiles[1])))
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

  INDEX_NAME,INDEX_DIMENSION = 'profiles', (768*2)

  #open connection with pinecone
  pinecone.init(
  api_key="your-pinecone_api-key",
  environment='your-pinecone-env', # find in console next to api key
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

    #remove all '\n's from data
    formattedData = formatData(profileData)

    embeddedData = []
    #load SBERT 'all-mpnet-base-v2' model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model.max_seq_length = 200
    
    #embed each experience and education text for each profile
    for i in range(len(profileURLS)):
      profileEmbed = [model.encode(formattedData[i][0]) , model.encode(formattedData[i][1])]
      embeddedData.append(profileEmbed)
      print("Embedded data:  " + profileURLS[i])

    #extend experinece embedding with education embedding
    extended = extendVec(embeddedData)

    #write data to file using pickle
    write_list(extended, "serializedData.txt")
    print("wrote data to file using pickle")

    #upload extended vectors to pinecone
    uploadVecToPinecone(extended[1:], profileURLS)
    print("Uploaded embeddings to pinecone except first profile")

    