Author
==============
God Bennett

Summary
==============
[Credit card fraud detection Ai layer by God Bennett.pdf](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/Credit%20Card%20Fraud%20Detection%20Ai%20Layer%20by%20God%20Bennett.pdf)

About
==============

Beginning on Tue 2/26/2019 (February 2019), this project sought to supplement current Fraud Guard method with neural network based code/method that I prepared/wrote, for the goal/purpose of [improved fraud detection by > 50%](http://news.mit.edu/2018/machine-learning-financial-credit-card-fraud-0920).

The current method of the Jamaican Bank's fraud detection though well structured, could be enhanced/supplemented by automatic means of credit card fraud detection, namely via the use of artificial intelligence. Artificial neural networks are quite general; there are neural networks that enable self-driving cars, while the same neural network types also enable disease diagnosis, language translation etc. 

The history of Ai has seen where expert systems with years of hand crafted rules/knowledge by experts, are enhanced considerably by automated systems that learn how to build rules. In some cases, hybrid systems have been constructed that make use of both learning ai, and rule-based ai.

Nowadays, most modern systems, including ones that [other banks like COK are using](https://www.fintechfutures.com/2018/10/smart-solution-gains-new-core-banking-tech-client-in-jamaica-cokcu/), make great use of the second wave of ai, namely statistical learning, or machine learning. The goal is to utilize the second wave of Ai, in tandem with current fraud guard systems, to greatly increase detection of frauds, while reducing the number of false positive detection.

As the bank gets more complex, we’ll reasonably need to use neural networks or some similar method to do fraud detection, because it is already hard for rule builders to keep up with fraud patterns with the current non-neural network based method, and neural network or similar methods capture more frauds, and minimizes the amount of transactions that are falsely detected as fraudulent, [by up to 54%.](http://news.mit.edu/2018/machine-learning-financial-credit-card-fraud-0920)

 * I proposed that the Jamaican bank shall seek to integrate a neural net based pipeline, using the credit card artificial neural network code prepared by myself that this repository refers to or similar.

Quick explanation of how this neural network works
==============
1. Take some transaction data, in the form of a database row for each transaction from the Jamaican Bank's Core Banking system.
2. Thousands of these transaction rows are labelled as 0 or 1 (not fraudulent or fraudulent) depending on fraudguard history. (Assumption: Data had been updated and properly labelled by Fraud Squad)
3. In training, expose the neural network to these labelled transactions, as the neural network learns what fraud and non-fraudulent looks like.
4. In testing aka inference (simulating when a single customer does a payment etc); expose neural network to unlabelled transactions. 
    * Neural network then produces a float value between 0 and 1 for each unlabelled transaction, where value closer to 1 indicates prediction of fraud, while closer to 0 indicates non-fraud.

See [this seminar lead by God Bennett, held for University Students, as well as a Jamaican Bank's team members, concerning basic artificial neural networks](https://github.com/JordanMicahBennett/Live-Agile-Artificial-Neural-Network-Programming-Sessions).



Original Code (55 lines)
==============
by Manuel on Kaggle: https://www.kaggle.com/manoloesparta/neural-network-accuracy-99-93



Modified Code (402 lines)
==============
by God Bennett/AdaLabs. 

* This code achieves an accuracy of ~95% on the Jamaican Bank's Core Banking system Dev transaction data.

 
Code Modification Description
==============
God wrote code to:

1. Perform individual testing of new transactions, (aka "online inferencing"), to simulate real-time processing of single transactions.
    * Note that online inferencing here, is not related to the internet. Online means per transaction database record neural network processing.
2. Perform crucial machine learning driven data imputation, since the Jamaican Bank's transaction data tends to have missing values.
3. Perform data visualization, including histograms, ... and confusion matrices that reflect accuracy of the model.
4. Perform lecun_normal initialization, instead of default "uniform" found in original code, for accuracy improvement, as the original code initializers did not work well. See pool of initializers at documentation site: https://keras.io/initializers/


Requirements
==============
Uses python 3.6, Requires keras, matplotlib, pandas, sklearn, numpy and tensorflow installations to python.



Installation
==============

1. Download [bennett_credit-card-fraud-detection_neural-network.py](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/bennett_credit-card-fraud-detection_neural-network%20%5B402_lines%5D.py) from this repository.
2. Put **[export_300k_v2_masked_JAMAICAN_BANK_Columns.csv](https://drive.google.com/file/d/1QuH-iWaFIOh1KtiyRy1BIHJ6rxReK2KR/view?usp=sharing)** in same directory as [bennett_credit-card-fraud-detection_neural-network.py](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/bennett_credit-card-fraud-detection_neural-network%20%5B402_lines%5D.py).
3. Install all python modules seen in [Requirements](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection#requirements).


Usage
==============
There are two ways to use this artificial neural network system:

1. Training and running.
    * **Train** the neural network (in about 3 minutes on an i7 cpu 8gb labtop) on the csv Jamaican Bank's transaction dev data. 
        * Note that most columns in the dataset are masked, and private to the Jamaican Bank. 
    * Run the trained neural network, and make some predictions.
        * **Training** is done by simply running the python file, and awaiting the neural network's processing for about 15 epochs.
            * A successful run will look [like this image](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/95.66%25_JAMAICAN_BANK_data_successful_run.png).
        * While making a prediction, take note of the "CARDFLAGFRAUD" column, which lables each transaction in dataset as 1 or 0 (where 1=fraud, 0=not fraudulent):
            * There are 299,999 records in dataset csv, and of those, the training process used the first 70%.
            * To really test the neural network, means to expose it to a record it didn't see in training.
            * Copy any record after cell 210,000 **(except for the last column which is the label)**. Records after 210,000 are outside of the "70%" training set.
            * Paste the copied record into python shell after neural network training ran, as parameter "newTransactionRecordString" from function [bennett_credit-card-fraud-detection_neural-network.py](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/bennett_credit-card-fraud-detection_neural-network%20%5B402_lines%5D.py)/ doOnlineInferenceOnRawRecord ( newTransactionRecordString ).
            * Take note of the result.
                * Eg a: Record A223999 is labelled 0, and neural net prediction is accurate at 0.029. (Closer to 0) See **data/notFraudulent_onlineInferenceOnRecord_A223999.png**.
				
                * Eg b: Record AY224046 is labelled 1, and [neural net prediction is accurate at 0.3381. (Closer to 1)  See **data/fraudulent_onlineInferenceOnRecord_AY224046.png**.
             

2. The quicker way: Running a pretrained model prepared by myself, that doesn't require training.
    * Download [JAMAICAN_BANK_load_saved_model_test.py](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/JAMAICAN_BANK_load_saved_model_test.py) from the data folder in this repository.
    * Download [95.66%_saved_JAMAICAN_BANK_neural_network_weights.h5](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/master/data/95.66%25_saved_JAMAICAN_BANK_neural_network_weights.h5) from the data folder of this repository.
    * Ensure the files above are in the directory of the csv file from the "[Installation](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/master/README.md#installation)" step of this repository.
    * To make predictions, do the same steps done in the training phase above, **particularly, starting from** "Run the trained neural network...".


Model accuracy in terms of confusion matrix
=============

It is important to guage how well an ai model is doing, in ai research and implementation.
"Confusion matrix" is a standard term in machine learning/artificial intelligence, that describes in this case:
1. The number of true positives aka correctly made predictions of detected fraud.
2. The number of true negatives aka correctly made predictions that indicate no fraud.
3. The number of false positives aka incorrectly made predictions that falsely indicate fraud.
4. The number of false negatives aka incorrectly made predictions that falsely indicate no fraud.
5. Overall accuray, as a function of the 4 items above.
    * **Total transactions** = false positives + false negatives + true positives + true negatives = (3708 + 197 + 533 + 85562) = 90,000
    * **Total correct predictions** or ‘true items’ = (true positives + true negatives )/Total transactions = (533 + 85562)/90,000 = 0.95661111111 ~ 95% accuracy
    * **Total correct predictions** or ‘false items’ = (false positives + false negatives)/Total = (3708 + 197)/90000 = (3708 + 197)/90000 ~ 0.043% inaccuracy



Invoking the function "showConfusionMatrix()" in [bennett_credit-card-fraud-detection_neural-network.py](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/bennett_credit-card-fraud-detection_neural-network%20%5B402_lines%5D.py) reveals the confusion matrix:

![Alt-Text](https://github.com/g0dEngineer/Ai-Credit-Card-Fraud-Detection/blob/main/95.66%25_JAMAICAN_BANK_data_confusion_matrix.png)

