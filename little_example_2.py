from mixture_of_unigram import mixture_of_unigram
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#let's create some simple documents first.
#we say first one belongs to topic 0. Second one to 1. third one to 2. fourth one to 0.
document_list=['apple banana apple cow apple apple banana apple','banana banana banana apple banana banana apple cow apple','down down down email email apple','apple apple apple apple banana apple cow apple']
topic_list=[0,1,2,0]

unlabeled_document_list=['banana banana banana apple cow apple cow banana banana email france']
test_document_list=['banana banana down email email france']

#let's calculate the word count. you can use some tools to do this. or do it yourself. we use CountVectorizer here.

vectorizer=CountVectorizer()
document_word_matrix=vectorizer.fit_transform(document_list).toarray()
show_word=vectorizer.get_feature_names()
unlabeled_word_matrix=vectorizer.transform(unlabeled_document_list).toarray()
test_word_matrix=vectorizer.transform(test_document_list).toarray()
print("show_word",show_word)
print("document word count",document_word_matrix)
print("unlabel word count",unlabeled_word_matrix)
print("test word count",test_word_matrix)
	
#initialization
#create our mixure of unigram model.
#show_word is a list of string(word).
#you can allow labeled document to be relabeled by specifying fix_labeled_doc as False. This could help if there are some incorrect labels.
#you can change prior(smoothing) by specifying a value for alpha, it should be greater or equal than 1.
model=mixture_of_unigram(show_word,topic_num=3,fix_labeled_doc=False,alpha=1.2)


#add labeled document
#labeled_word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
#topic is a list(len=word_matrix.shape[0]) of int(topic) the document belongs in.
model.add_labeled_doc(document_word_matrix,topic_list)

#When there is only labeled data. We run EM algorithm one time and it will have the best solution.
model.train(iteration=1)


#add unlabeled document
#unlabeled_word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
#now there are labeled and unlabeled documents in our model.
#You can train them together. Augment training data by unlabeled data.
model.add_unlabeled_doc(unlabeled_word_matrix)

#We can run EM algorithm more times this time.
model.train(iteration=100)

#predict which topic does these document belongs to.
print("test predict topic",model.predict(test_word_matrix).argmax(axis=1))

#if you want to get a raw distribution over topic for each document for further analysis.
print("test predict topic distribution",model.predict(test_word_matrix))

#show 10 words that best represent each topic.
print("topic",model.get_topic(show_word=2))


#you can also access the distribution over topic of the documents in the model.
print("labeled data topic distribution",model.labeled_doc2topic)
print("unlabeled data topic distribution",model.doc2topic) #this one for unlabeled document.

#get the topic these document belongs to is the same.
print("labeled data topic",model.labeled_doc2topic.argmax(axis=1))
print("unlabeled data topic",model.doc2topic.argmax(axis=1)) #this one for unlabeled document.

