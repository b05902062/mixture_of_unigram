from mixture_of_unigram import mixture_of_unigram
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#let's create some simple documents first.
#we say first one belongs to topic 0. Second one to 1. third one to 2. fourth one to 0.
lab_doc=['apple banana apple cow apple apple banana apple','banana banana banana apple banana banana apple cow apple','down down down email email apple','apple apple apple apple banana apple cow apple']
topic_list=[0,1,2,0]

test_doc=['banana banana down email email france']

#let's calculate the word count. you can use some tools to do this. or do it yourself. we use CountVectorizer here.

vectorizer=CountVectorizer()
#v_lab_doc and v_test_doc are both 2d array specifying the count of each word in each document. They are of size #_of_document*len(# of words).
#Each row correspond to a document. Each column corresponds to the word in the same column in show_word.
v_lab_doc=vectorizer.fit_transform(lab_doc).toarray()
show_word=vectorizer.get_feature_names()
v_test_doc=vectorizer.transform(test_doc).toarray()
print("show_word\n",show_word)
print("document word count\n",v_lab_doc)
print("test word count\n",v_test_doc)
	
#initialization
#create our mixure of unigram model.
#show_word is a list of string(word).
model=mixture_of_unigram(show_word,topic_num=3)


#add labeled document
#v_lab_doc is a 2d array specifying the count of each word in each document. It is of size #_of_document*len(# of different words).
#topic is a list(len=v_lab_doc.shape[0]) of int(topic index) the document belongs in.
model.add_labeled_doc(v_lab_doc,topic_list)

#When there is only labeled data. We run EM algorithm one time and it will have the best solution.
model.train(iteration=1)


#use trained model to predict which topic do new documents belong to as below.
#v_test_doc is also a 2d array specifying the count of each word in each document. It is of size #_of_documents_in_test_doc*len(show_word).
print("predict topic",model.predict(v_test_doc).argmax(axis=1))


#if you want to get a raw distribution over topic for each document for further analysis.
print("test predict topic distribution\n",model.predict(v_test_doc))


#show 2 words that best represent each topic. The earlier a word appears the more descriptive the word is about that topic.
print("topic\n",model.get_topic(show_word=2))

