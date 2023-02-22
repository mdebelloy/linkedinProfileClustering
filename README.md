# LinkedinProfileClustering
This code is used with an additional Linkedin profile .txt list with links to the profiles that should be scraped
# profileScrapingToPinecone.py 
This code scrapes the list of Linkedin profiles for their education and experiences tabs, embeds the first 200 words of each using SBERT and the all-mpnet-base-v2 model. It then extends one embedding with another to get the total embedding for the profile, and uploads it to Pinecone.io. The first profile in the list is not uploaded to be used for querying as shown below
# queryPinecone.py
This code takes the first profile from the list and queries Pinecone for the closest 3 other profiles in Eucledian distance, which gives an output like this:
```
  The 3 closest profiles to  https://www.linkedin.com/in/filip-dvorak-n1/  are: 
  
  {'matches': [{'id': 'maximillian-burton-637b214/',
                'score': 2.04834867,
                'values': []},
               {'id': 'himanshu-gupta-83477a17/',
                'score': 2.10669565,
                'values': []},
               {'id': 'oskarhjertonsson/', 'score': 2.13094831, 'values': []}],
   'namespace': ''}
```

# plotCluster.py
This code takes the embedded dataset, reduces its dimensionality from 1536 to 30 using PCA, then to 2 using T-SNE. The data is then plotted, and colored according to a K-means clustering. The number associated to each point shows what profile it represents:
![ClustersV1](https://user-images.githubusercontent.com/11065853/220770986-72ae1034-ddba-4369-bb5c-5582335b26d0.png)

```
1  -->  https://www.linkedin.com/in/filip-dvorak-n1/
2  -->  https://www.linkedin.com/in/colbylepore/
3  -->  https://www.linkedin.com/in/venkat-mattela-ph-d-6762684/
4  -->  https://www.linkedin.com/in/joreyramer/
5  -->  https://www.linkedin.com/in/gsuri/
6  -->  https://www.linkedin.com/in/ryzhaya/
7  -->  https://www.linkedin.com/in/achuthanand-ravi-b341ab59/
8  -->  https://www.linkedin.com/in/mltan/
9  -->  https://www.linkedin.com/in/tim-zheng/
10  -->  https://www.linkedin.com/in/charleymoore/
11  -->  https://www.linkedin.com/in/brettadcock/
12  -->  https://www.linkedin.com/in/ellenrudolph/
13  -->  https://www.linkedin.com/in/himanshu-gupta-83477a17/
14  -->  https://www.linkedin.com/in/alexandrwang/
15  -->  https://www.linkedin.com/in/toyand/
16  -->  https://www.linkedin.com/in/jcarrharris/
17  -->  https://www.linkedin.com/in/timhyoung/
18  -->  https://www.linkedin.com/in/johnfreddyvega/
19  -->  https://www.linkedin.com/in/paul-hamadeh-5a09599/
20  -->  https://www.linkedin.com/in/kylevogt/
21  -->  https://www.linkedin.com/in/maximillian-burton-637b214/
22  -->  https://www.linkedin.com/in/samzaid/
23  -->  https://www.linkedin.com/in/oskarhjertonsson/
24  -->  https://www.linkedin.com/in/gans20m/
25  -->  https://www.linkedin.com/in/daveko/
26  -->  https://www.linkedin.com/in/michaelactonsmith/
27  -->  https://www.linkedin.com/in/aaron-lown-2a239111/
28  -->  https://www.linkedin.com/in/markingsley/
29  -->  https://www.linkedin.com/in/briancollins3/
30  -->  https://www.linkedin.com/in/jessicawalsh1/
31  -->  https://www.linkedin.com/in/mrjamesmartin/
```
