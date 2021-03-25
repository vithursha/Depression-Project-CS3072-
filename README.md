This project aims to detect depression in short text using machine learning algorithms. For simplicity, the steps involved in this project have been
broken down into individual folders in the 'Depression-Project-CS3072-'. The folders are as follows: 

# 1. Pre-processing and EDA 
This folder includes the source code for the pre-processing and EDA steps. The files include: 
	1.1 Pre-Processing Fiorela Dataset
	- This file consists of the source code for the pre-processing steps carried out on the (Fiorela, 2020)dataset. Steps include expanding contractions, 
	removing punctuations and removing stop words before saving the processed dataset as a new CSV file called 'DepressionDataFinal2Processed.csv'.  
		1.1.1 Pre-Processing Fiorela Dataset and Stem Lem - This file consists of the source code for the stemming and lemmatisation steps in 
		addition to the pre-processing steps carried out on the (Fiorela, 2020) dataset. The processed CSV file was saved as 
		'DepressionDataFinal2ProcessedStemLem.csv'.
	1.2 Pre-Processing Romero Dataset
	- This file consists of the source code for the pre-processing steps carried out on the (Romero, 2019) dataset. Steps include expanding contractions, 
	removing punctuations and removing stop words before saving the processed dataset as a new CSV file called 'RomeroDatasetProcessed.csv'.  
	1.3 EDA - WordCloud Fiorela and Romero Dataset
	- This file includes the source code for the wordcloud generated on the datasets. It also includes two pie charts representing the number of depressed 
	and non-depressed tweets in both datasets. These steps aided in concluding with the suitable dataset.  
	1.4 EDA - Feature Engineering, N-Grams, POS Tags and Bivariate Analysis on Fiorela Dataset
	- After finalising the dataset, more EDA was performed on the Fiorela dataset. These steps included calculating average word lengths between 
	depressed and non-depressed tweets and comparing n-grams and POS Tags. 
	Reference:
	tutorial video: https://www.youtube.com/watch?v=HVBk2Ge_Q98
	contractions: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
	removing punctuations: https://www.pluralsight.com/guides/importance-of-text-pre-processing
	wordcloud: https://www.analyticsvidhya.com/blog/2020/10/word-cloud-or-tag-cloud-in-python/
	piechart: https://pythonspot.com/matplotlib-pie-chart/
	eda: https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/
	
# 2. Machine Learning Classifiers
This folder includes the source code to train and test machine learning classifiers. The files include: 
	2.1 Machine Learning Classifiers of Depression Dataset with Stem and Lem 
	- As part of the pre-processing step, 'DepressionDataFinal2Processed.csv' and 'DepressionDataFinal2ProcessedStemLem.csv' were used to train machine 
	learning classifiers on to discover if stemming and lemmatisation steps were essential during the pre-processing steps. To evaluate the results, 
	classification reports were printed to show the results of standard evaluation metrics such as accuracy, precision, recall and f1_score. It was 
	concluded that stemming and lemmatisation were not required during the pre-processing step as the results did not improve much. 
	2.2 Machine Learning Classifiers Run and Evaluated on Fiorela Dataset 										
	- This source code involves building the chosen machine learning classifiers, training them on the processed training dataset, before predicting the
	classes on the test dataset and evaluating the predicted results. For the classifiers to be able to understand the tweets, the vocabulary in the tweets
	was converted into a dictionary of words and their unique index using the CountVectorizer() class provided by sklearn. To make the evaluation fair, 
	the classifiers were also tested on a 20% test dataset called 'Test Dataset.csv' which was created when running neural network models. 
	Reference:
	ML classifier: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
	CountVectorizer() class: https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/ 
	xgb: https://xgboost.readthedocs.io/en/latest/python/python_api.html
	confusion matrix: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

# 3. Finetune Neural Network Models
- This folder contains 3 files with the source code for the exhaustive grid search which was performed to find optimal hyperparameters on recurrent neural 
network,long-short term memory and gated recurrent unit model. Several grid search parameters were explored before concluding with the most efficient models.
Reference:
Grid search: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/ 

# 4. Neural Network Models 
- This folder contains 4 subfolders for the 4 different data representation techniques used and the 4 subfolders consist of 3 files each corresponding to the
individual neural network models used in this project. The files are: 
	4.1 Tokenizer() class 
	- In this folder, the files used the Tokenizer() class to convert the text into numerical form to be fed into the neural network models. The files were: 
		4.1.1 Recurrent Neural Network using Tokenizer() class  
		- In this file, the dataset was first converted into the numerical form using the CountVectorizer() class to be fed into the neural 
		network model. Followed by that the RNN model was built with the finetuned parameters and compiled before running it. Then the model was 
		evaluated on the x_test data (created during the 60% train, 20% validation and 20% test split) and evaluation metrics and confusion matrix were 
		computed. This test dataset was saved as 'Test Dataset.csv' to be used to evaluate all the models. 
		4.1.2 Long-Short Term Memory using Tokenizer() class
		- In this file, the data was first converted into numerical form. Followed by that the LSTM model was built with the finetuned parameters
		 and compiled before running it. Finally, the model was evaluated on x_test data and evaluation metrics and confusion matrix were computed. 
		4.1.3 Gated Recurrent Unit using Tokenizer() class
		- In this file, the data was first converted into numerical form. Followed by that the GRU model was built with the finetuned parameters
		 and compiled before running it. Finally, the model was evaluated on x_test data and evaluation metrics and confusion matrix were computed. 
	4.2 Word2Vec on Depression Corpus
	- In this folder, the 3 neural networks that used the word2vec embedding trained on the depression dataset, can be found as 3 separate files. After the 
	data is converted into numerical form, the word2vec embedding model was built and trained on the dataset. Followed by that the words and their 
	vectors were saved as a dictionary called the 'embedding_word2vec.txt' file. Then an embedding weight matrix was built, and the vectors of the dictionary
	were added to the matrix. Finally, the embedding matrix was set as the weight parameter in the Embedding layer of the neural network models, before building, 
	compiling, testing and evaluating the model. 
	4.3 Word2Vec on Google's Pre-train Corpus
	- In this folder, the 3 neural networks that used the pretrained Google’s Word2Vec embedding model, can be found as 3 separate files. After the 
	CountVectorizer() class converted the data into numerical form, the pretrained Google’s Word2Vec embedding model, renamed 'GoogleVec.bin' for simplification, 
	was finetuned on the depression dataset by creating an embedding weight matrix, and copying the embedding vectors for the words from
	the dataset from the Google's word2Vec model to the matrix. Finally, the embedding matrix was set as the weight parameter in the Embedding layer 
	of the neural network models, before building, compiling testing and evaluating the model. 
	4.4 GloVe Embedding
	- In this folder, the 3 neural networks that used the GloVe embedding layer, can be found as 3 separate files. After converting the data into 
	numerical form, the GloVe embedding saved as a text file called 'glovetwitter27B200d.txt' was opened and a weight matrix was created and populated
	with the vectors for the words from the dataset from the GloVe embedding dictionary and the matrix was set as the weight parameter in the Embedding
	layer of the neural network model, before building, compiling testing and evaluating the model. 	
	Reference:
	data representation techniques (Tokenizer() class, Word2Vec and GloVe embedding): Learning with Python by Francois Chollet, Implement neural networks
	with Keras on Theano and TensorFlow by Sujit Pal et al. 
	spam detection example: https://towardsdatascience.com/nlp-spam-detection-in-sms-text-data-using-deep-learning-b8632db85cc8
	recurrent layers: https://keras.io/api/layers/recurrent_layers/

# 5. BERT Model 
This folder contains the source code for the BERT model. 
	5.1 BERT model with ktrain wrapper
	- This file includes the source code for the BERT model built using the ktrain wrapper and its evaluation steps. This includes the evaluation metrics
 	that were computer after building and training the BERT model on the depression dataset and the confusion matrix. Since running the model is 
	time-consuming, the model was saved in the 'BERT' folder to be used during the evaluation stage. 
	5.2 Evaluation of BERT model on Test Dataset
	- In this file, the saved BERT model was loaded and tested on the Test Dataset that was created. The evaluation metrics and the confusion matrix 
	were also computed to compare the performance with the other models. Additionally, the 21 misclassified tweets were printed and compared to 
	critically analyse the model's performance. 
	Reference:
	Tutorial on ktrain: https://github.com/amaiya/ktrain
	Finetune tutorial: https://towardsdatascience.com/ktrain-a-lightweight-wrapper-for-keras-to-help-train-neural-networks-82851ba889c	

thumbs_up.jpeg - This image was used when generating the wordcloud for non-depressive tweets 
thumbs_down.jpeg - This image was used when generating the word cloud for the depressive tweets 
DepressionDataFinal2.csv - This CSV file is the unprocessed Fiorela dataset.
DepressionDataFinal2Processed.csv - This CSV file is the processed Fiorela dataset after performing the pre-processing steps not including stemming and lemmatisation steps. 
DepressionDataFinal2ProcessedStemLem.csv -  This CSV file is the processed Fiorela dataset after performing the pre-processing steps including stemming and lemmatisation steps.  
RomeroDataset.csv - This CSV file is the unprocessed Romero dataset.
RomeroDatasetProcessed.csv - This CSV file is the processed Romero dataset.
Test_Dataset.csv - This CSV file consists of 1763 tweets (20% of the dataset) and is the test dataset that was used to evaluate the models.  
embedding_word2vec.txt - word2vec embedding created when training the embedding on depression dataset 

# Additional Files 
https://nlp.stanford.edu/projects/glove/ - The GloVe model was downloaded from this link (download 'glove.twitter.27B.zip' file and use 200 dimensional model)
https://code.google.com/archive/p/word2vec/#! - The pre-trained Word2Vec model trained on Google News dataset can be found of this link (download 
' GoogleNews-vectors-negative300.bin.gz')

# Setup 
To run this file, download the Depression Project folder and upload it to a suitable virtual notebook environment. To access the csv files, change the path to the correct file paths. 
To run the neural network models with the Word2Vec and GloVe embedding, download the embeddings from the link provided above and change the path to the documents accordingly. 
