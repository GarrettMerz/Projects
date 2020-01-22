# Kaggles

# Some Kaggle projects to practice various data science skills.

# 'Butterflies' is the Kaggle "Moths and Butterflies" fieldguide challenge:
# https://www.kaggle.com/c/fieldguide-challenge-moths-and-butterflies/data
# The goal of the challenge is to identify the butterfly or moth from a number of species based on the images provided.


# Input data is in both the json annotation files provided in the link and the two tarballs at 
# https://s3.amazonaws.com/fieldguide-fgvc2019/training.tar
# https://s3.amazonaws.com/fieldguide-fgvc2019/testing.tar
# each of which contains an image of a butterfly or moth that is 600 px in its largest dimension

# Currently, I have produced a simple convolutional network which runs using Tensorflow 2.1 / Cuda 10.1 on my laptop RTX 2060 gpu. Eventually, I would like to train a fully-convolutional network to avoid padding the input images.

# 'Mushrooms' is the Kaggle "Mushroom Identification" fieldguide challenge:
# https://www.kaggle.com/uciml/mushroom-classification
# The goal of the challenge is to develop a tool to determine the edibility of a mushroom using these samples taken from 23 species of the Agaricus and Lepiota families.

# Data is in the csv file in the directory, each row corresponding to one mushroom with the following categorical attributes:
#    classes: edible=e, poisonous=p
#    cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
#    cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
#    cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
#    bruises: bruises=t,no=f
#    odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
#    gill-attachment: attached=a,descending=d,free=f,notched=n
#    gill-spacing: close=c,crowded=w,distant=d
#    gill-size: broad=b,narrow=n
#    gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
#    stalk-shape: enlarging=e,tapering=t
#    stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
#    stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#    stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
#    stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#    stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#    veil-type: partial=p,universal=u
#    veil-color: brown=n,orange=o,white=w,yellow=y
#    ring-number: none=n,one=o,two=t
#    ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
#    spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
#    population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
#    habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

# Data is not split into train/test/validation samples in the csv; this is done so at the classifier stage.
