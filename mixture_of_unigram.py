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

class mixture_of_unigram():
	def __init__(self,word_matrix,show_word,topic_num=10):
		#word_matrix is the one returned by sklearn.vectorizer()
		self.topic_num=topic_num
		self.word_matrix=word_matrix
		self.show_word=show_word
		self.doc_num=word_matrix.shape[0]
		self.word_num=word_matrix.shape[1]
		
		#randomly initialize topic to word distribution
		self.topic2word=np.random.uniform(0,1,[self.topic_num,self.word_num])
		self.topic2word=self.topic2word/(self.topic2word.sum(axis=1,keepdims=True)+10**-200)
		#randomly initialize topic sampling probability
		self.topic=np.random.uniform(0,1,[self.topic_num])
		self.topic=self.topic/(self.topic.sum()+10**-200)


	def train(self,iteration=10):
		for i in range(iteration):
			doc2topic=self.__expectation(self.word_matrix)
			self.__maximization(doc2topic)

	def __maximization(self,doc2topic):
		#doc2topic size doc_num*topic

		assert((self.doc_num+self.topic_num)!=0)
		self.topic=(doc2topic.sum(axis=0)+1)/(self.doc_num+self.topic_num)
		doc2word=self.word_matrix.reshape(self.doc_num,-1,self.word_num)
		#size topic*word
		self.topic2word=1+(doc2topic.reshape(self.doc_num,self.topic_num,-1)*doc2word).sum(axis=0)
		self.topic2word=self.topic2word/(self.topic2word.sum(axis=1,keepdims=True))

	def __expectation(self,word_matrix):
		
		#calculate the probability of documents belonging to a topic conditioned on current self.topic and self.topic2word.

		assert(word_matrix.shape[1]==self.word_num)
		#initalize to 1.
		doc2topic=np.full([word_matrix.shape[0],self.topic_num],1.)

		#More efficient. It is the same as the lines quoted below.
		doc2topic=doc2topic*self.topic
		doc2word=word_matrix.reshape(word_matrix.shape[0],self.word_num,-1)
		topic2word=self.topic2word.transpose([1,0]).reshape(-1,self.word_num,self.topic_num)
		doc2word=np.power(topic2word,doc2word)
		
		#normalize to improve floating point precision
		doc2word=doc2word/(doc2word.mean(axis=2,keepdims=True))
		
		doc2word=np.prod(doc2word,axis=1)
		doc2topic*=doc2word
		doc2topic=doc2topic/(doc2topic.sum(axis=1,keepdims=True)+10**-200)
		
		"""
		for i_doc,doc in enumerate(self.word_matrix):
			for i_topic in range(self.topic_num):
				doc2topic[i_doc][i_topic]*=self.model2topic[i_topic]
				for i_word,word in enumerate(doc):
					doc2topic[i_doc][i_topic]*=self.topic2word[i_topic][word]
		doc2topic=doc2topic/doc2topic.sum(dim=1)
		"""
		
		#return size word_matrix.shape[0]*topic
		return doc2topic

	def inference(self,word_matrix):
		doc2topic=self.__expectation(word_matrix)
		return doc2topic.argmax(axis=1)

	def get_topic(self,word_num=3):
		topic=[]
		for i in range(self.topic_num):
			topic.append([self.show_word[o] for o in self.topic2word[i].argsort()[:-word_num-1:-1]])
		return topic


def main():
	#preprocess your file according to your file format. You can also perform other prepocessing technique in advance.
	word_matrix,show_word=read_file("../aaai.json")
	
	#word_matrix is a 2d array specifying the count of each word in each document. It is of size #_of_string*len(words). show_word is a list of string(word).
	model=mixture_of_unigram(word_matrix,show_word,topic_num=10)
	model.train(iteration=100)

	for i in range(10):
		print(model.get_topic(word_num=10)[i])
	
	print(model.inference(word_matrix)[:10])

if __name__=="__main__":
	main()
