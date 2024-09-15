# Code Cell
from google.colab import drive
drive.mount('/content/drive')

# Code Cell
#Installation commands
!unzip '/content/drive/MyDrive/stanford-parser-full-2018-02-27.zip'
!pip install svgling

# Code Cell
#Import statements

import os
from nltk.parse.stanford import StanfordParser
from nltk import ParentedTree, Tree
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re

# Code Cell
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Code Cell
#Setting java path
java_path = "/usr/lib/jvm/java-11-openjdk-amd64/bin/java"
os.environ['JAVAHOME'] = java_path

# Code Cell
sp = StanfordParser(path_to_jar='stanford-parser-full-2018-02-27/stanford-parser.jar',
                    path_to_models_jar='stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar')

# Code Cell
stopwords_set = set(['a', 'an', 'the', 'is','to','The','in','of','us'])

# Code Cell
def convert_isl(parsetree):
  dict = {}
  parenttree= ParentedTree.convert(parsetree)
  for sub in parenttree.subtrees():
    dict[sub.treeposition()]=0
  #"----------------------------------------------"
  isltree=Tree('ROOT',[])
  i=0
  for sub in parenttree.subtrees():
    if(sub.label()=="NP" and dict[sub.treeposition()]==0 and dict[sub.parent().treeposition()]==0):
      dict[sub.treeposition()]=1
      isltree.insert(i,sub)
      i=i+1

    if(sub.label()=="VP" or sub.label()=="PRP"):
      for sub2 in sub.subtrees():
        if((sub2.label()=="NP" or sub2.label()=='PRP') and dict[sub2.treeposition()]==0 and dict[sub2.parent().treeposition()]==0):
          dict[sub2.treeposition()]=1
          isltree.insert(i,sub2)
          i=i+1

  for sub in parenttree.subtrees():
    for sub2 in sub.subtrees():
      if(len(sub2.leaves())==1 and dict[sub2.treeposition()]==0 and dict[sub2.parent().treeposition()]==0):
          dict[sub2.treeposition()]=1
          isltree.insert(i,sub2)
          i=i+1

  return isltree

# Code Cell
def text_to_isl(sentence):
  pattern = r'[^\w\s]'

# Remove the punctuation marks from the sentence
  sentence = re.sub(pattern, '', sentence)
  englishtree=[tree for tree in sp.parse(sentence.split())]
  parsetree=englishtree[0]
  #print(parsetree)
  isl_tree = convert_isl(parsetree)
  words=parsetree.leaves()
  lemmatizer = WordNetLemmatizer()
  ps = PorterStemmer()
  lemmatized_words=[]
  for w in words:
    #w = ps.stem(w)
   lemmatized_words.append(lemmatizer.lemmatize(w))
   islsentence = ""
   for w in lemmatized_words:
    if w not in stopwords_set:
      islsentence+=w
      islsentence+=" "
  #islsentence = clean(words)
  islsentence = islsentence.lower()
  isltree=[tree for tree in sp.parse(islsentence.split())]
  return islsentence

# Code Cell
!pip install moviepy
!pip install pytube

# Code Cell
from pytube import YouTube
def get_yt(link,path):
  yt = YouTube(link)
  yt.streams.filter(file_extension="mp4").get_by_resolution("360p").download(path)

# Code Cell
from moviepy.editor import *
import os

def cut_vid(filename,yt_name,start_min,start_sec,end_min,end_sec):
  clip = VideoFileClip(os.path.join(yt_path,yt_name+'.mp4'))
  clip1 = clip.subclip((start_min,start_sec),(end_min,end_sec))
  clip1.write_videofile(os.path.join('/content/NLP_dataset',filename+'.mp4'),codec='libx264')

# Code Cell
!mkdir NLP_dataset
!mkdir yt

# Code Cell
root_path = '/content/NLP_dataset'
yt_path = '/content/yt'

# Code Cell
import pandas as pd

# Code Cell
NLP_videos = pd.read_csv('/content/drive/MyDrive/NLP_videos.csv')

# Code Cell
import pandas as pd
from IPython.display import HTML
from base64 import b64encode

# Code Cell
NLP_videos = pd.read_csv('/content/drive/MyDrive/NLP_videos.csv')
NLP_videos.head()

# Code Cell
def text_to_vid(input_text):
  NLP_videos = pd.read_csv('/content/drive/MyDrive/NLP_videos.csv')
  root_path = '/content/NLP_dataset'
  yt_path = '/content/yt'
  videos = []
  clips=[]
  sentence = text_to_isl(input_text)
  print(sentence)
  words = sentence.split()
  for i in words:
    if (NLP_videos['Name'].eq(i)).any():
      idx = NLP_videos.index[NLP_videos['Name'] == i].tolist()
      get_yt(NLP_videos['Link'].iloc[idx[0]],yt_path)
      cut_vid(i,NLP_videos['yt_name'].iloc[idx[0]],NLP_videos['start_min'].iloc[idx[0]],NLP_videos['start_sec'].iloc[idx[0]],NLP_videos['end_min'].iloc[idx[0]],NLP_videos['end_sec'].iloc[idx[0]])
      videos.append(os.path.join(root_path,i+'.mp4'))
    else:
      for letter in i:
        idx = NLP_videos.index[NLP_videos['Name'] == letter].tolist()
        #print(letter,idx)
        get_yt(NLP_videos['Link'].iloc[idx[0]],yt_path)
        cut_vid(letter,NLP_videos['yt_name'].iloc[idx[0]],NLP_videos['start_min'].iloc[idx[0]],NLP_videos['start_sec'].iloc[idx[0]],NLP_videos['end_min'].iloc[idx[0]],NLP_videos['end_sec'].iloc[idx[0]])
        videos.append(os.path.join(root_path,letter+'.mp4'))
  for i in videos:
    clip = VideoFileClip(i)
    clips.append(clip)

  final = concatenate_videoclips(clips, method="compose")
  final.write_videofile("merged.mp4")

# Markdown Cell
"""
# Speech To Text
"""

# Code Cell
!pip install SpeechRecognition
!pip install pyaudio
!pip install ffmpeg-python

# Code Cell
import speech_recognition as sr
r = sr.Recognizer()

# Code Cell
AUDIO_FILE = ("/content/NLP_test (2).wav")

with sr.AudioFile(AUDIO_FILE) as source:
    #reads the audio file. Here we use record instead of
    #listen
    audio = r.record(source)
speech=r.recognize_google(audio)

# Code Cell
print(speech)

# Markdown Cell
"""
# Speech to Video
"""

# Code Cell
text_to_vid(speech)

# Code Cell
mp4 = open('merged.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
  <video width=400 controls>
      <source src="%s" type="video/mp4">
  </video>
  """ % data_url)

