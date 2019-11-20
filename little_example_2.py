from mixture_of_unigram import mixture_of_unigram
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#let's create some simple documents first.
#we say first one belongs to topic 0. Second one to 1. third one to 2. fourth one to 0.
lab_doc=['apple banana apple cow apple apple banana apple','banana banana banana apple banana banana apple cow apple','down down down email email apple','apple apple apple apple banana apple cow apple']
topic_list=[0,1,2,0]

unlab_doc=['banana banana banana apple cow apple cow banana banana email france']


more_lab_doc=['cow cow banana cow apple banana banana','apple apple cow banana france apple france']
more_topic_list=[1,0]
more_unlab_doc=['banana banana banana apple cow cow banana banana email france']

test_doc=['banana banana down email email france']

#let's calculate the word count. you can use some tools to do this. or do it yourself. we use CountVectorizer here.

vectorizer=CountVectorizer()
#documents are all 2d array specifying the count of each word in each document. They are of size #_of_document*len(# of words). Each row correspond to a document. Each column corresponds to the word in the same column in show_word.
v_lab_doc=vectorizer.fit_transform(lab_doc).toarray()
show_word=vectorizer.get_feature_names()
v_unlab_doc=vectorizer.transform(unlab_doc).toarray()
v_more_lab_doc=vectorizer.transform(more_lab_doc).toarray()
v_more_unlab_doc=vectorizer.transform(more_unlab_doc).toarray()
v_test_doc=vectorizer.transform(test_doc).toarray()
print("show_word\n",show_word)
print("document word count\n",v_lab_doc)
print("unlabel word count\n",v_unlab_doc)
print("test word count\n",v_test_doc)
	
#initialization
#create our mixure of unigram model.
#show_word is a list of string(word).
#you can allow labeled document to be relabeled by specifying fix_labeled_doc as False. This could help if there are some incorrect labels.
#you can change prior(smoothing) by specifying a value for alpha, it should be greater or equal than 1. Default value is 2.
model=mixture_of_unigram(show_word,topic_num=3,fix_labeled_doc=False,alpha=1.2)


#add labeled document
#v_lab_doc is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
#topic is a list(len=v_lab_doc.shape[0]) of int(topic index) the document belongs in.
model.add_labeled_doc(v_lab_doc,topic_list)

#When there is only labeled data. We run EM algorithm one time and it will have the best solution.
model.train(iteration=1)


#add unlabeled document
#v_unlab_doc is a 2d array specifying the count of each word in each document. It is of size #_of_documents*len(show_word).
#now there are labeled and unlabeled documents in our model. They will be trained together. Augment training data by unlabeled data.
#training without labeled data is also possible. But it may suffer from bad initialization. Maybe combining unsupervised training with some other ideas like running it in a hierarchical way could help.
model.add_unlabeled_doc(v_unlab_doc)

#We can run EM algorithm more times this time.
model.train(iteration=100)

#use trained model to predict which topic do new documents belong to as below.
#v_test_doc is also a 2d array specifying the count of each word in each document. It is of size #_of_documents_in_test_doc*len(show_word).
print("test predict topic\n",model.predict(v_test_doc).argmax(axis=1))

#if you want to get a raw distribution over topic for each document for further analysis.
print("test predict topic distribution\n",model.predict(v_test_doc))

#we can continue to add more labeled or unlabeled data into our model.
model.add_labeled_doc(v_more_lab_doc,more_topic_list)
model.add_unlabeled_doc(v_more_unlab_doc)

model.train(iteration=100)


print("test predict topic after adding more data\n",model.predict(v_test_doc).argmax(axis=1))
print("test predict topic distribution after adding more data\n",model.predict(v_test_doc))

#show 10 words that best represent each topic.
print("topic\n",model.get_topic(show_word=2))


#you can also access the distribution over topic of the documents in the model.
print("labeled data topic distribution\n",model.labeled_doc2topic)
print("unlabeled data topic distribution\n",model.doc2topic) #this one for unlabeled document.

#get the topic these document belongs to is the same.
print("labeled data topic\n",model.labeled_doc2topic.argmax(axis=1))
print("unlabeled data topic\n",model.doc2topic.argmax(axis=1)) #this one for unlabeled document.

