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
population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
Data is not split into train/test/validation samples in the csv; this is done so at the classifier stage.

#RollerDerby

When I'm not coding data science projects, I enjoy announcing my friends' roller derby bouts (currently with the Kalamazoo Derby Darlins).
This is my attempt to use data science tools to extract some meaningful quantities from roller derby statistics. 
In the sport, each team has a line of four "blockers" that attempt to hold back the opposing team's "jammer"; breaking through a "blocker" line earns the team four points.
Multiple passes through the blocker line will occur over the course of one two-minute "jam", however, the "jammer" that is in the lead has the option to "call off the jam" before the alloted two minutes is up, allowing teams to swap out players and return to a fixed starting line.

This project is an attempt at using a Graph Convolutional Neural Nets (GCNN) autoencoder to determine the optimal composition of a blocker line. It will generally follow the methodology of this paper: https://arxiv.org/pdf/1805.03285.pdf


SOME PROJECTS I HAVE NOT YET STARTED, BUT THAT WILL GO HERE:

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

from the Vatican Archives as a good training set for exploring handwriting segmentation.

#Cribbing Pseudo-autoencoder

The Liber Primus is a text released in 2014 by internet cipherpunk collective Cicada 3301 that has yet to be decrypted (https://uncovering-cicada.fandom.com/wiki/Liber_Primus). Frequency analysis of the text indicates it is likely encoded with some kind of autokey-variant cipher. The text has spacing, which means that, if this spacing is correct, it can be "cribbed"- plaintext words can be compared against the ciphertext in order to determine the algorithm. Since it is likely that the function is nonlinear, this is my attempt to use a reinforcement learning/autoencoder type algorithm to learn the cipher map implicitly.
