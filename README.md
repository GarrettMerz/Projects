# Projects

# Some Kaggle projects to practice various data science skills.

#ButterflyNet
This is my convnet solution to the Kaggle "Moths and Butterflies" fieldguide challenge:
https://www.kaggle.com/c/fieldguide-challenge-moths-and-butterflies/data
The goal of the challenge is to identify the butterfly or moth from a number of species based on the images provided.


Input data is in both the json annotation files provided in the link and the two tarballs at 
https://s3.amazonaws.com/fieldguide-fgvc2019/training.tar
https://s3.amazonaws.com/fieldguide-fgvc2019/testing.tar
each of which contains an image of a butterfly or moth that is 600 px in its largest dimension.

Currently, I have produced a simple convolutional network which runs using Tensorflow 2.1 / Cuda 10.1 on my laptop RTX 2060 gpu.
Eventually, I would like to train a fully-convolutional network to avoid padding the input images, currently, this project serves as more of a hardware test.

#Mushrooms
This the Kaggle "Mushroom Identification" fieldguide challenge:
 https://www.kaggle.com/uciml/mushroom-classification

The goal of the challenge is to develop a tool to determine the edibility of a mushroom using these samples taken from 23 species of the Agaricus and Lepiota families.
Data is in the csv file in the directory, each row corresponding to one mushroom with the following categorical attributes:
 
classes: edible=e, poisonous=p
cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises: bruises=t,no=f
odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
gill-attachment: attached=a,descending=d,free=f,notched=n
gill-spacing: close=c,crowded=w,distant=d
gill-size: broad=b,narrow=n
gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
stalk-shape: enlarging=e,tapering=t
stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
veil-type: partial=p,universal=u
veil-color: brown=n,orange=o,white=w,yellow=y
ring-number: none=n,one=o,two=t
ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

1

# Projects

2

​

3

# Some Kaggle projects to practice various data science skills.

4

​

5

#ButterflyNet

6

This is my convnet solution to the Kaggle "Moths and Butterflies" fieldguide challenge:

7

https://www.kaggle.com/c/fieldguide-challenge-moths-and-butterflies/data

8

The goal of the challenge is to identify the butterfly or moth from a number of species based on the images provided.

9

​

10

​

11

Input data is in both the json annotation files provided in the link and the two tarballs at 

12

https://s3.amazonaws.com/fieldguide-fgvc2019/training.tar

13

https://s3.amazonaws.com/fieldguide-fgvc2019/testing.tar

14

each of which contains an image of a butterfly or moth that is 600 px in its largest dimension.

15

​

16

Currently, I have produced a simple convolutional network which runs using Tensorflow 2.1 / Cuda 10.1 on my laptop RTX 2060 gpu.

17

Eventually, I would like to train a fully-convolutional network to avoid padding the input images, currently, this project serves as more of a hardware test.

18

​

19

#Mushrooms

20

This the Kaggle "Mushroom Identification" fieldguide challenge:

21

 https://www.kaggle.com/uciml/mushroom-classification

22

​

23

The goal of the challenge is to develop a tool to determine the edibility of a mushroom using these samples taken from 23 species of the Agaricus and Lepiota families.

24

Data is in the csv file in the directory, each row corresponding to one mushroom with the following categorical attributes:

25

 

26

classes: edible=e, poisonous=p

27

cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

28

cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

29

cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

30

bruises: bruises=t,no=f

31

odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

32

gill-attachment: attached=a,descending=d,free=f,notched=n

33

gill-spacing: close=c,crowded=w,distant=d

34

gill-size: broad=b,narrow=n

35

gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

36

stalk-shape: enlarging=e,tapering=t

37

stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

38

stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

39

stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

40

stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

41

stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

42

veil-type: partial=p,universal=u

43

veil-color: brown=n,orange=o,white=w,yellow=y

44

ring-number: none=n,one=o,two=t

45

ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

@GarrettMerz

population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
Data is not split into train/test/validation samples in the csv; this is done so at the classifier stage.

#RollerDerby

When I'm not coding data science projects, I enjoy announcing my friends' roller derby bouts (currently with the Kalamazoo Derby Darlins).
This is my attempt to use data science tools to extract some meaningful quantities from roller derby statistics. 
In the sport, each team has a line of four "blockers" that attempt to hold back the opposing team's "jammer"; breaking through a "blocker" line earns the team four points.
Multiple passes through the blocker line will occur over the course of one two-minute "jam", however, the "jammer" that is in the lead has the option to "call off the jam" before the alloted two minutes is up, allowing teams to swap out players and return to a fixed starting line.


Questions I am interested in include:
-which combinations of players perform the best together?
-when is it better to run a jam for the full two minutes?
-which "jammers" are best against which blocker lines?


#Voynich
The Voynich Manuscript is one of the most interesting problems in cryptography. Written sometime in Europe in the 1400s, it is a text written in an unknown script corresponding to no known language. 
The text is accompanied by a series of illustrations, depicting everything from unidentifiable plants to constellations.
Zipf's Law, entropy, and frequency analysis studies have shown that the text is not random, and does in fact possess some informational content. Furthermore, analysis of handwriting and vocabulary have shown that the text was written by two scholars, each with a slightly different style (one more verbose than the other). Additionally, recent handwriting-analysis work has shown that the text was written by at least five different scribes.
Currently, I am investigating several questions.

-Can I use clustering methods to extract the two sets of authorship? What about the different scribes?
-Can I use topic analysis to choose the words most representative of each section?
-How do different characters and combinations behave statistically?
-How accurate is the establshed "EVA" transcription?

#In Codice Ratio
One of the problems facing analyses of historical documents is the unavailability (and unreliability!) of digital transcriptions. Digital transcription tools often have special difficulty with context-dependent sentence parsing, especially with Latin 'sigla' or scribal abbreviations, which can mean different things based on their locations in words. Sayre's Paradox is rampant in these datasets- in order to be understood, a word must be segmented into characters, but in order for a cursive word to be correctly segmented into characters, it must be understood! This project will attempt to utilize the dataset here:  

http://www.inf.uniroma3.it/db/icr/datasets.html 

from the Vatican Archives as a good training ground for exploring handwriting segmentation.

(thought- can I use some kind of adversarial net here(or at least nested ones?)? train one NN to segment based on word-piece similarity to labeled train-set characters, train another based on the words arising from the characters chosen using this segmentation scheme vs. language corpus? is this what ICR does already?)
