# Machine Learning Awesome List

Maintainer - [sdukshis](https://github.com/sdukshis)

## Table of Contents
- [Code](#code)
- [Courses](#courses)
- [Conferences](#conferences)
- [Papers](#papers)
- [Thesis](#thesis)
- [Books](#books)
- [Journals](#journals)
- [Datasets](#datasets)
- [Blogposts](#blogposts)
- [Competitions](#competitions)

## Code
* Neural Networks
    * [Theano](http://www.deeplearning.net/software/theano/) - Python
    * [TensorFlow](https://www.tensorflow.org/) - Python, C++
    * [Lasagne](http://lasagne.readthedocs.io/en/latest/) - Python
    * [Keras](http://keras.io/) - Python
    * [mxnet](https://github.com/dmlc/mxnet) - Python, C++, R, Scala, Julia, Go, Matlab, js
    * [Caffe](http://caffe.berkeleyvision.org/) - Python, C++
    * [cuDNN](https://developer.nvidia.com/cudnn) - C++
* [scikit-learn](http://scikit-learn.org/) - Python
* [Rodeo](https://www.yhat.com/) - Python
* [BigARTM](http://bigartm.org/) - Python

## Courses
* Machine Learning
    * [Машинное обучение (К.В.Воронцов)](http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%28%D0%BA%D1%83%D1%80%D1%81_%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D0%B9%2C_%D0%9A.%D0%92.%D0%92%D0%BE%D1%80%D0%BE%D0%BD%D1%86%D0%BE%D0%B2%29)
    * [Введение в машинное обучение (К.В. Воронцов)](https://www.coursera.org/learn/vvedenie-mashinnoe-obuchenie)
    * [Machine Learning (Andrew Ng)](https://www.coursera.org/learn/machine-learning)
* Neural Networks
    * [CS231n Convolutional Neural Networks for Visual Recognition by Andrej Karpathy](http://cs231n.github.io/)
    * [Глубинное обучение (курс лекций)](http://www.machinelearning.ru/wiki/index.php?title=Глубинное_обучение_%28курс_лекций%29)
    
## Conferences
 * Neural Information Processing Systems (NIPS) - [site](http://nips.cc)
 * International Conference on Learning Representations (ICLR) - [site](http://www.iclr.cc)
 * International Conference on Machine Learning (ICML) - [site](http://www.icml.cc)

## Papers
* Neural Networks
    * Deep learning
        * Learning multiple layers of representation, Geoffrey Hinton, 2007 - [Paper](http://www.cs.toronto.edu/~fritz/absps/tics.pdf)
        * Learning Deep Architectures for AT, Yoshua Bengio - [Paper](http://people.cs.pitt.edu/~huynv/research/deep-nets/Learning%20Deep%20Architectures%20for%20AI.pdf)
        * Dropout:  A Simple Way to Prevent Neural Networks from Overfitting, N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov - [Paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
        * Reducing the Dimensionality of Data with Neural Networks, G. E. Hinton* and R. R. Salakhutdinov - [Paper](http://www.cs.toronto.edu/~hinton/science.pdf)
        * A theoretical framework for deep transfer learning, T. Galanti, L. Wolf, and T. Hazan - [Paper](http://imaiai.oxfordjournals.org/content/early/2016/04/28/imaiai.iaw008.full.pdf)
        * Comparative Study of Deep Learning Software Frameworks, Soheil Bahrampour, Naveen Ramakrishnan, Lukas Schott, Moh
ak Shah - [Paper](http://arxiv.org/abs/1511.06435)
        * Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Sergey Ioffe, Christian Szegedy - [Paper](Sergey Ioffe, Christian Szegedy)
    * Recurrent Networks
        * Long Short-Term Memory, S. Hochreiter, J. Schmidhuber, 1997 - [Paper](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        * Long Short Term Memory Networks for Anomaly Detection in Time Series, P. Malhotra, L. Vig, G. Shroff, P. Agarwal, 2015 - [Paper](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf)
        * A Clockwork RNN, Jan Koutnik, Klaus Greff, Faustino Gomez, Jürgen Schmidhuber, 2014 - [Paper](http://arxiv.org/pdf/1402.3511v1.pdf)
        * Sequence Labelling in Structured Domains with Hierarchical Recurrent Neural Networks, Santiago Fernandez, Alex Graves, Jurgen Schmidhuber, 2007 - [Paper](ftp://ftp.idsia.ch/pub/juergen/IJCAI07sequence.pdf)
        * Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks, Alex Graves, Santiago Fernandez, Faustino Gomez, Jurgen Schmidhuber, 2006 - [Paper](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf)
        * Learning Long-Term Dependencies with Gradient Descent is Difficult, Y. Bengio, P. Simard, and P. Frasconi, 1994 - [Paper](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf)
        * Character-Aware Neural Language Models, Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush - [Paper](http://www.people.fas.harvard.edu/%7Eyoonkim/data/aaai_2016.pdf)
    * Convolutional Neural Networks
        * Time Series Classification Using Multi-Channels Deep Convolutional Neural Networks, Yi Zheng, Qi Liu, Enhong Chen, Yong Ge, and J Lean Zhao, 2014 - [Paper](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf)
        * Convolutional Networks for Images, Speech, and Time-Series, Yann Lecun, Yoshua Bengio - [Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-bengio-95a.pdf)
        * Understanding Convolutional Neural Networks, Jayanth Koushik - [Paper](https://arxiv.org/pdf/1605.09081v1.pdf)
        * Deep learning, Yann LeCun, Yoshua Bengio, andGeoffrey Hinton - [Paper](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
        * Leaning longer memory in recurrent neural networks, Tomas Mikolov, Armand Joulin, Sumit Chopra, Michael Mathieu & Marc’Aurelio Ranzato - [Paper](https://arxiv.org/pdf/1412.7753.pdf)
        * Recurrent neural network regularization, Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals - [Paper](https://arxiv.org/pdf/1409.2329v5.pdf)
        * Learning to forget: continual prediction with LSTM, Felix Gers,  Jurgen Schmidhuber, 1999 - [Paper](https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf)
        * Unitary Evolution Recurrent Neural Networks, Martin Arjovsky, Amar Shah, Yoshua Bengio, 2015 - [Paper](https://arxiv.org/abs/1511.06464)
        
* Time Series Anomaly Detection
    * SAX
        * HOT SAX: Finding the Most Unusual Time Series Subsequence: Algorithms and Applications, Eamonn Keogh, Jessica Lin, Ada Fu, 2005 - [Paper](http://www.cs.ucr.edu/~eamonn/discords/HOT%20SAX%20%20long-ver.pdf), [Materials](http://www.cs.ucr.edu/~eamonn/discords/)
    * LSTM
        * LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection, Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, Gautam Shroff, 2016 - [Paper](https://drive.google.com/file/d/0B8Dg3PBX90KNQWRwMElkVkQ0aFgzZGpzOGQtUU5DeWZYUlVV/view)
    * Transfer learning
        * Transfer Representation-Learning for Anomaly Detection, Jerone T. A. Andrews, Thomas Tanay, Edward J. Morton, Lewis D. Griffin, 2016 - [Paper](https://drive.google.com/file/d/0B8Dg3PBX90KNeFROU3BDT1ZhTXlSV3Rsb3JfVWNTWkpLTUhJ/view)
    * Anomaly Detection Based on Sensor Data in Petroleum Industry Applications, Luis Martí,1, Nayat Sanchez-Pi, José Manuel Molina, and Ana Cristina Bicharra Garcia - [Paper](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4367333/)
    * Anomaly detection in aircraft data using recurrent nueral networks (RNN), Anvardh Nanduri, Lance Sherry - [Paper](http://catsr.ite.gmu.edu/pubs/ICNS_2016_AnomalyDetectionRNN_01042015.pdf)
    * Bayesian Online Changepoint Detection, Ryan Prescott Adams, David J.C. MacKay - [Paper](http://hips.seas.harvard.edu/files/adams-changepoint-tr-2007.pdf)

* Clustering
    * Consistent Algorithms for Clustering Time Series, Azadeh Khaleghi, Daniil Ryabko, Jeremie Mary, Philippe Preux, 2016 - [Paper](http://jmlr.csail.mit.edu/papers/volume17/khaleghi16a/khaleghi16a.pdf)

## Thesis
* [Statistical Language Models based on Neural Networks by Tomas Mikolov](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
* [Time Series Prediction Using Neural Networks by Karol Kuna, 2015](http://is.muni.cz/th/410446/fi_b/thesis.pdf)
* [TRAINING RECURRENT NEURAL NETWORKS by Ilya Sutskever, 2013](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
* [Anomaly Detection of Time Series](http://conservancy.umn.edu/bitstream/handle/11299/92985/?sequence=1)
* [Unsupervised Anomaly Detection in Sequences Using Long Short Term Memory Recurrent Neural Networks](http://mars.gmu.edu/handle/1920/10250)

## Books
* Machine Learning: A Bayesian and Optimization Perspective by Sergios Theodoridis - [Amazon](http://www.amazon.com/Machine-Learning-Optimization-Perspective-Developers/dp/0128015225/ref=pd_sim_14_3?ie=UTF8&dpID=51vPYhMsTvL&dpSrc=sims&preST=_AC_UL160_SR128%2C160_&refRID=12KNBY7VJ04SMYBXKQQ7), [Safari](https://www.safaribooksonline.com/library/view/machine-learning/9780128015223/)
* Fundamentals of Deep Learning, Nikhil Buduma - [Safari](https://www.safaribooksonline.com/library/view/fundamentals-of-deep/9781491925607/)
* Rank Based Anomaly Detection Algorithms - [Book](http://surface.syr.edu/cgi/viewcontent.cgi?article=1335&context=eecs_etd)
* Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville - [Book](http://www.deeplearningbook.org/)
* Foundations of Data Science, Avrim Blum, John Hopcroft and Ravindran Kannan - [Book](http://www.cs.cornell.edu/jeh/book2016June9.pdf)
 
## Journals
* [Journal of Machine Learning Research](http://jmlr.csail.mit.edu/)
* [Machine Learning and Data Analysis ](http://jmlda.org/)
* [Machine Learning](http://www.springer.com/computer/ai/journal/10994)
* [International Journal of Machine Learning and Cybernetics](http://www.springer.com/engineering/computational+intelligence+and+complexity/journal/13042/PSE)
* [Data Mining and Knowledge Discovery](http://www.springer.com/computer/database+management+%26+information+retrieval/journal/10618)
* [Intelligent Data Analysis](http://www.iospress.nl/journal/intelligent-data-analysis/)
* [Интеллектуальные системы](http://www.intsys.msu.ru/magazine/)
* [Artificial Intelligence](https://www.elsevier.com/journals/artificial-intelligence/0004-3702#description)
* [Artificial Intelligence Review](http://www.springer.com/computer/ai/journal/10462)
* [Engineering Applications of Artificial Intelligence](https://www.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976#description)

## Datasets
* [PhysioBank](http://www.physionet.org/cgi-bin/atm/ATM)
* [UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/)
* [NASA Shuttle Valve Data](http://cs.fit.edu/~pkc/nasa/data/)
* [Yahoo Labeled Anomaly Detection Dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)
* [Awesome Public Datasets](https://github.com/caesar0301/awesome-public-datasets)
* [The Numenta Anomaly Benchmark Competition For Real-Time Anomaly Detection](http://numenta.org/nab/)
* [An archive of datasets distributed with R](https://vincentarelbundock.github.io/Rdatasets/datasets.html)
* [List of datasets for machine learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)
* [Disk Defect Data](https://c3.nasa.gov/dashlink/resources/314/)

## Blogposts
* [Estimating Rainfall From Weather Radar Readings Using Recurrent Neural Networks](http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/)
* [Theoretical Motivations for Deep Learning](http://rinuboney.github.io/2015/10/18/theoretical-motivations-deep-learning.html)
* [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/)
* [Common Objects in Context](http://mscoco.org/)
* [Neural Network Architectures](https://culurciello.github.io/tech/2016/06/04/nets.html)
* [Building powerful image classification models using very little data](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
* [Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)
* [Written Memories: Understanding, Deriving and Extending the LSTM](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html)
* [Time Series Prediction With Deep Learning in Keras](http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/)

## Competitions
* [Senior Data Science: Safe Aging with SPHERE](https://www.drivendata.org/competitions/42/page/71/)
* [ML Boot Camp](http://mlbootcamp.ru)
