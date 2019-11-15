import json
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import re
from mixture_of_unigram import mixture_of_unigram

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
	word_matrix,show_word=read_file("./aaai.json")
	

	#initialization
	#show_word is a list of string(word).
	model=mixture_of_unigram(show_word,topic_num=10,fix_labeled_doc=False,alpha=1.1)


	#add unlabeled document
	#unlabeled_word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_sentences*len(show_word).
	#This is a unsupervised classification, which would not be good. Maybe running it in a hirarchical way may produce some good result.
	model.add_unlabeled_doc(word_matrix)


	#We can run EM algorithm more times this time.
	model.train(iteration=30)

	#show 10 words that best represent each topic.
	print(model.get_topic(show_word=10))
	
	#predict which topic does these document belongs to.
	print(model.predict(word_matrix).argmax(axis=1)[:10])

if __name__=="__main__":
	main()
