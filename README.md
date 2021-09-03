# Projects

# Some Kaggle projects to practice various ML skills.

#ButterflyNet
This is my convnet solution to the Kaggle "Moths and Butterflies" fieldguide challenge:
https://www.kaggle.com/c/fieldguide-challenge-moths-and-butterflies/data
The goal of the challenge is to identify the butterfly or moth from a number of species based on the images provided.


Input data is in both the json annotation files provided in the link and the two tarballs at 
https://s3.amazonaws.com/fieldguide-fgvc2019/training.tar
https://s3.amazonaws.com/fieldguide-fgvc2019/testing.tar
each of which contains an image of a butterfly or moth that is 600 px in its largest dimension.

Currently, I have produced a (very slim) ResNet which runs using Keras/ Tensorflow 2.5 / Cuda 11 on my laptop RTX 2060 gpu. Eventually, I'd like to get this running on Google CoLab so I can make it bigger (and train faster).
Performance on this isn't stellar, but it's a functional ResNet model!

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
This is my attempt at performing an end-to-end ML project, including data gathering, by myself, from scratch (no kaggle!).

In the sport, each team has a line of four "blockers" that attempt to hold back the opposing team's "jammer"; breaking through a "blocker" line earns the team four points. Multiple passes through the blocker line will occur over the course of one two-minute "jam", however, the "jammer" that is in the lead has the option to "call off the jam" before the alloted two minutes is up, allowing teams to swap out players and return to a fixed starting line.

This project is an attempt at using a modified Structural Deep Neighbor Embedding graph autoencoder to build a recommender system that determines the optimal composition of a blocker line. It will generally follow the methodology of this paper: https://arxiv.org/pdf/1805.03285.pdf.

I'll compare the custom 'Teammate' autoencoder against both standard SDNE and the HOPE embedding model. In the future, I may return to this project to explore whether GCN or GAT models improve performance further. 

The recommender system here works well- it has a low MSE, and is able to successfully predict performance on the validation set.

#TweetyBERT

https://www.kaggle.com/kazanova/sentiment140

This is an attempt at using DistilBERT to do binary "positive vs negative" sentiment analysis on a large corpus of labeled Twitter data.
This is mostly a way for me to get familiar with BERT models and some NLP best practices.
Ultimately, I see an accuracy of greater than 85% on the validation set- pretrained BERT is a powerful model that runs amazingly quickly!


