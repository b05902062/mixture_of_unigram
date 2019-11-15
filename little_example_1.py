from mixture_of_unigram import mixture_of_unigram
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#let's create some simple documents first.
#we say first one belongs to topic 0. Second one to 1. third one to 2. fourth one to 0.
document_list=['apple banana apple cow apple apple banana apple','banana banana banana apple banana banana apple cow apple','down down down email email apple','apple apple apple apple banana apple cow apple']
topic_list=[0,1,2,0]

test_document_list=['banana banana banana apple cow apple cow banana banana email france']

#let's calculate the word count. you can use some tools to do this. or do it yourself. we use CountVectorizer here.

vectorizer=CountVectorizer()
document_word_matrix=vectorizer.fit_transform(document_list).toarray()
show_word=vectorizer.get_feature_names()
test_word_matrix=vectorizer.transform(test_document_list).toarray()
print("corresponding word",show_word)
print("document word count",document_word_matrix)
print("test word count",test_word_matrix)
	
#initialization
#create our mixure of unigram model.
#show_word is a list of string(word).
model=mixture_of_unigram(show_word,topic_num=3)


#add labeled document
#labeled_word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
#topic is a list(len=word_matrix.shape[0]) of int(topic) the document belongs in.
model.add_labeled_doc(document_word_matrix,topic_list)

#When there is only labeled data. We run EM algorithm one time and it will have the best solution.
model.train(iteration=1)


#test_word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
#use trained model to predict which topic does new document belongs to.
print("predict topic",model.predict(test_word_matrix).argmax(axis=1))

#if you want to get a raw distribution over topic for each document for further analysis.
print("predict topic distribution",model.predict(test_word_matrix))

#show 1 words that best represent each topic.
print("topic",model.get_topic(show_word=2))

