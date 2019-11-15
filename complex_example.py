import json
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import re
def read_file(file_name):
	with open(file_name,"r") as f:
	  aaai=json.load(f)

	title_list=[]
	abstract_list=[]
	for i,title_and_abstract in aaai.items():
		title_list+=[title_and_abstract["title"]]
		abstract_list+=[title_and_abstract['abstract']]

	abstract_list=[unicodedata.normalize("NFC",i) for i in abstract_list]
	abstract_list=[i.lower() for i in abstract_list]
	abstract_list=[re.sub("[,.'?!:;]"," ",i) for i in abstract_list]
	abstract_list=[re.sub("[ ]+"," ",i) for i in abstract_list]
	p_stemmer = PorterStemmer()
	abstract_list = [" ".join([p_stemmer.stem(t) for t in i.split()]) for i in abstract_list]

	#print("print",abstract_list[:10])
	vectorizer = CountVectorizer(stop_words="english",min_df=5,max_df=200)
	word_matrix = vectorizer.fit_transform(abstract_list)
	show_word = vectorizer.get_feature_names()
	print("# of words",len(show_word))
	return word_matrix.toarray(),show_word



def main():
	#preprocess your file according to your file format. You can also perform other prepocessing technique in advance.
	word_matrix,show_word=read_file("../aaai.json")
	

	#initialization
	#show_word is a list of string(word).
	model=mixture_of_unigram(show_word,topic_num=10)

	#add labeled document
	#labeled_word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
	#topic is a list(len=word_matrix.shape[0]) of int(topic) the document belongs in.
	model.add_labeled_doc(labeled_word_matrix,topic)

	#When there is only labeled data. We run EM algorithm one time and it will have the best solution.
	model.train(iteration=1)

	#add unlabeled document
	#unlabeled_word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
	#now there are labeled and unlabeled documents in our model.
	model.add_unlabeled_doc(unlabeled_word_matrix)


	#We can run EM algorithm more times this time.
	model.train(iteration=100)

	#show 10 words that best represent each topic.
	print(model.get_topic(show_word=10))
	
	#predict which topic does these document belongs to.
	print(model.predict(word_matrix).argmax(axis=1))
	
	#if you want to get a raw distribution over topic for each document for further analysis.
	print(model.predict(word_matrix))

	#you can also access the distribution over topic of the documents in the model.
	print(model.labeled_doc2topic)
	print(model.doc2topic) #this one for unlabeled document.
	
	#get the topic these document belongs to is the same.
	print(model.labeled_doc2topic.argmax(axis=1))
	print(model.doc2topic.argmax(axis=1)) #this one for unlabeled document.


if __name__=="__main__":
	main()
