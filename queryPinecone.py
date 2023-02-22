import pinecone
import pickle

#query pinecone for similar profiles given an embedded vector
def queryPinecone(vector):

  INDEX_NAME,INDEX_DIMENSION = 'profiles', (768*2)

 #open connection with pinecone
  pinecone.init(
  api_key="your-pinecone_api-key",
  environment='your-pinecone-env', # find in console next to api key
  )

  index = pinecone.Index(index_name=INDEX_NAME)
  
  #return 3 closest profiles
  return index.query(top_k=3, include_values=False, include_metadata=False, vector=vector.tolist())

#read URLs from text file
def getURLS(file):
  file = open(file, 'r')
  return file.read().splitlines()

if __name__ == "__main__":
  #get profile link of unused profile
  profileURLS = getURLS("profilesList.txt")

  #use the fact that the first profile wasn't uploaded to pinecone
  extended = pickle.load( open( "serializedData.txt", "rb" ) )
  
  print("The 3 closest profiles to ", profileURLS[0], " are: ")
  print(queryPinecone(extended[0]))