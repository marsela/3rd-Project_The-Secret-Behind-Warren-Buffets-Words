# The Secret Behind Warren Buffet's Words

## Using an NLP algorithm on Berkshire Hathway's Annual Report

The goal of this project is to take the words of legendary investor Warren Buffet and try to create a prediction model. We will try to predict the performance of Warren Buffet's company, Berkshire Hathway, by using the words written in the annual report. The main assumption is that within the annual report are conversations about the past, present and future of Berkshire Hathway. For a more accurate prediction model, we could have only used the portion of the annual report that pertains to Berkshire's future but we will use the entire annual report and hope the algorithm can detect all those words that are important.

Before we can run out prediction algorithms, we must first process the annual reports so that it can be useable by our models. The annual reports will under go the following steps: 

* **Step 1:**  Scrape Annual Reports from Berkshire Hathaway website

* **Step 2:**  Format and create the target feature

* **Step 3:**  Clean the text by removing stop words and stemming other words

We will grab the annual reports from Berkshire's corporate website. Unfortunately, the website only has annual reports dating back to 1977. If our prediction produce usable results, we would further add to the accuracy of the model by adding those missing reports. For the target feature, we will us the "Berkshire’s Performance vs. the S&P 500" table from this year's annual letter to shareholder. We will create both a classification model and a prediction model. The models will attempt to answer the following questions respectively.

1. Will Berkshire Hathway(BRK) Out Perform The S&P 500? (Classification Model)
2. What is the predicted stock performance of Berkshire? (Regression Model)

The end result of the processing the annual report into a "bag of words" creates a dataset where each word becomes a binary feature of the model. We can use these features without any limitations as to the type of Machine Learning algorithm that could run. Our only issue is that we have more features than we have instances. This is can happen when we are dealing with Natural Language Processing. 


###  Book Value versus Market Value

In the table, "Berkshire’s Performance vs. the S&P 500"  we can see the performance of Berkshire Hathway's as compared to the S&P 500. The table further breaks down the company's value using its Book value and Market Value. We use both of these values as the target feature for our prediction model. 

* **[Section 1: Prediction Model - Market Value](Section1_MarketValue.ipynb)** — The Market value is the price an asset would fetch in the marketplace or what a company would sell for if someone were to buy the entire company today. The Market value can fluctuate a great deal over periods of time and is substantially influenced by the business cycle. Market value for a firm may diverge significantly from book value or shareholders’ equity. 

* **[Section 2: Prediction Model - Book Value](Section2_BookValue.ipynb)** — Book value is the net asset value of a company, calculated as total assets minus intangible assets (patents, goodwill) and liabilities. It is also the value at which the asset is carried on a balance sheet and calculated by taking the cost of an asset minus the accumulated depreciation. When compared to the company's market value, book value can indicate whether a stock is under- or overpriced. Since the Book Value fluctuates less than Market Value, the model should be more accurate predicting book value.

* **[Section 3: Word Cloud](WordCloud.ipynb)** — The following is a Word Cloud generated from selected annual reports. 


###  Summary of Machine Learning Algorithms



* **Linear and Logistic Regression** — We will use the linear and logistic regression as our baseline model for our regression and classification model respectively. 


* **Support Vector Regression(SVR)** — We will use a Support Vector Regression to create nonlinear models. Normally we need to scale our features before we can use our Support Vector Regression but since we are working with the "bag of words" all of our features will have a value of 1 or 0 and therefore is already scaled. We will choose a Gaussian kernel for our Support Vector.   


* **Naive Bayes** — Naive Bayes Classifiers are a family of simple probabilistic classifier based on applying Bayes' theorem without the requirement of independence assumptions between the features. The components of Bayes theorem; posterior probability, likelihood, prior probability and marginal likelihood are all used. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem.


* **Random Forest Regression** — Random forest classifier creates a set of decision trees from the randomly selected subset of the training set. It then aggregates the votes from different decision trees to decide the final class of the test object. Basic parameters to Random Forest Classifier can be the total number of trees to be generated and decision tree related parameters like minimum split criteria etc. Random decision forests correct for decision trees' habit of overfitting to their training by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. 


* **Artifical Neural Network** — We will use a standard neural network with backward propagation. In order to run this section of the notebook, we had to install Google's Deep Learning platform, [TensorFlow](https://www.tensorflow.org/) (acting as a backend) and [Keras](https://keras.io/) (acting as a frontend). We had two hidden layers in our neural network with 1000 nodes in each layer. This is reasonable since we had approximately 8000 features to feed into the neural network.  We added a 10% drop out after every hidden layer to minimize the effects of overfitting. 


* **XgBoost** — XgBoost is one of the most popular models in machine learning. It is also the most powerful implementation of gradient boosting. One of the major advantages of Xgboost besides having high performance and fast execution speed is that you can keep the interpretation of the original problem.  We were unable to do a K-fold cross-validation with our limited computational power.


You can also read a [Warren Report](report.pdf) which summarizes the implementation as well as the methodology of the whole project.


Even though we are running an Artifical Neural Network with thousands of features, it is not necessary to run the algorithm on a GPU. This is due to the binary nature of the each of the features, which represent whether a word is present or not. Using a GPU to run the Neural Network would, of course, speed things up, but it is not necessary. Instructions to set up the GPU are written below



## Requirements

### Dataset

The dataset consists of all of the annual reports available for free at [Berkshire Hathaway's website](http://www.berkshirehathaway.com/letters/letters.html). Most of the annual reports are in the form of html webpages while the reports from 2001 onwards are in pdf form. The annual reports in html form were scraped using Python's "beautiful soup" package.  For those reports that are in pdf form, we can to manual convert them to text files using Adobe Reader text conversion. Once these files were converted, we appended them to the rest of the annual report. 


### Software

This project uses the following software (if the version number is omitted, the latest version is recommended):


* **Python stack**: python 3.5.3, numpy, scipy, sklearn, pandas, matplotlib, h5py.
* **Neural Network**: multi-threaded xgboost should be compiled, xgboost python package is also required.
* **Deep Learning stack**: CUDA 8.0.44, cuDNN 5.1, TensorFlow 1.1.0 Keras 2.0.6


## Guide to running this project

### Option 1 - Setting up Desktop to run  Nvidia's GeForce 770 (This project)

**Step 1. Install necessary drivers to use GPU**
The desktop is running Windows 7 with the following installs:
If you need the C++ compiler, you can download it here (**[C++ Compiler](http://landinghub.visualstudio.com/visual-cpp-build-tools)**) 

* **cuda toolkit -** https://developer.nvidia.com/cuda-toolkit -  The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler and a runtime library to deploy your application.

* **Nvidia Drivers -** http://www.nvidia.com/Download/index.aspx

* **cuDNN 7 -** https://developer.nvidia.com/cudnn - cuDNN is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

**Step 2. Install DeepLearning Packages using Conda**
* **Theano -** ' `conda install -c conda-forge theano`

* **Tensorflow GPU-** 

 `conda create -n tensorflow python=3.5`
 
 `activate tensorflow`
 
 `pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-win_amd64.whl`


* **Keras -** `conda install -c conda-forge keras`

### Option 2 - Using AWS instances

**Step 1. Launch EC2 instance**
The cheaper option to run the project is to use EC2 AWS instances:

* `c4.8xlarge` CPU optimized instance for Feature Selection calculations (best for Part 2).
* `EC2 g2.2xlarge` GPU optimized instance for MLP and ensemble calculations (best for Part 3). If you run an Ireland-based spot instance, the price will be about $0.65 per hour or you can use "Spot Instances" to help reduce cost.

* For more detail instructions view the following link : http://markus.com/install-theano-on-aws/

Please make sure you run Ubuntu 14.04. For Ireland region you can use this AMI: **ami-ed82e39e**. Also, add 30 GB of EBS volume 

**Step 2. Clone this project **

`sudo apt-get install git`

`cd ~; git clone https://github.com/volak4/Zillow.git`

`cd Zillow`

`activate tensorflow`
