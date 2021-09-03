SOME IDEAS FOR PROJECTS I HAVE NOT YET STARTED, BUT THAT WILL LIKELY GO HERE:

#Voynich
The Voynich Manuscript is one of the most interesting problems in cryptography. Written sometime in Europe in the 1400s, it is a text written in an unknown script corresponding to no known language.
The text is accompanied by a series of illustrations, depicting everything from unidentifiable plants to constellations.
Zipf's Law, entropy, and frequency analysis studies have shown that the text is not random, and does in fact possess some informational content. Furthermore, analysis of handwriting and vocabulary have
shown that the text was written by two scholars, each with a slightly different style (one more verbose than the other). Additionally, recent handwriting-analysis work has shown that the text
was written by at least five different scribes. Currently, I am investigating several questions.

-Can I use clustering methods to extract the two sets of authorship? What about the different scribes?
-Can I use topic analysis to choose the words most representative of each section?
-What does self-attention tell us about the Voynich, at word and character level?
-How do different characters and combinations behave statistically?
-How accurate is the establshed "EVA" transcription?

#In Codice Ratio
One of the problems facing analyses of historical documents is the unavailability (and unreliability!) of digital transcriptions. Digital transcription tools often have special difficulty with 
context-dependent sentence parsing, especially with Latin 'sigla' or scribal abbreviations, which can mean different things based on their locations in words. Sayre's Paradox is rampant in these 
datasets- in order to be understood, a word must be segmented into characters, but in order for a cursive word to be correctly segmented into characters, it must be understood! 
This project will attempt to utilize the dataset here:

http://www.inf.uniroma3.it/db/icr/datasets.html

from the Vatican Archives as a good training set for exploring handwriting segmentation.

#Cribbing Pseudo-autoencoder

The Liber Primus is a text released in 2014 by internet cipherpunk collective Cicada 3301 that has yet to be decrypted (https://uncovering-cicada.fandom.com/wiki/Liber_Primus). 
Frequency analysis of the text indicates it is likely encoded with some kind of autokey-variant cipher. The text has spacing, which means that, if this spacing is correct, it can be "cribbed"- 
plaintext words can be compared against the ciphertext in order to determine the algorithm. Since it is likely that the function is nonlinear, this is my attempt to use a reinforcement 
learning/autoencoder type algorithm to learn the cipher map implicitly.
