# Big Data

## Small Data VS Big Data

**Small Data**: refers to sets of data that are small enough to be understood and manipulated by a human. Small Data may not have the same scale as Big Data, but it can be more accessible and targeted. The data is normally structured and of high quality. Small Data can be used to generate useful and direct insights and is often employed in strategic decisions and analysis

**Big Data**: is a collection of large and complex data sets,that cannot be processed by databases or applicationstraditional processing.
We can define the concept of Big Data as being sets of dataextremely broad and that, for this reason, need tools specially prepared to deal with large volumes, speed and variety, so that any and all information available in the data can be found, analyzed and used in a timely manner.

Below is a table of differences between Small Data and Big Data, taken from : [geeksforgeeks.org](https://www.geeksforgeeks.org/difference-between-small-data-and-big-data/)

| Feature |	Small Data |	Big Data |
|------------| -----------| ---------- |
| Variety |	Data is typically structured and uniform | Data is often unstructured and heterogeneous |
| Veracity |	Data is generally high quality and reliable | Data quality and reliability can vary widely |
| Processing |	Data can often be processed on a single machine or in-memory | Data requires distributed processing frameworks such as MapReduce or Spark |
| Technology | Traditional | Modern |
| Analytics |	Traditional statistical techniques can be used to analyze data | Advanced analytics techniques such as machine learning are often required |
| Collection |	Generally, it is obtained in an organized manner than is inserted into the database | The Big Data collection is done by using pipelines having queues like AWS Kinesis or Google Pub / Sub to balance high-speed data |
| Volume |	Data in the range of tens or hundreds of Gigabytes | Size of Data is more than Terabytes |
| Analysis Areas | Data marts(Analysts) | Clusters(Data Scientists), Data marts(Analysts) |
| Quality |	Contains less noise as data is less collected in a controlled manner | Usually, the quality of data is not guaranteed |
| Processing |	It requires batch-oriented processing pipelines | It has both batch and stream processing pipelines |
| Database | SQL | NoSQL |
| Velocity |	A regulated and constant flow of data, data aggregation is slow | Data arrives at extremely high speeds, large volumes of data aggregation in a short time |
| Structure | Structured data in tabular format with fixed schema(Relational) | Numerous variety of data set including tabular data, text, audio, images, video, logs, JSON etc.(Non Relational) |
| Scalability | They are usually vertically scaled | They are mostly based on horizontally scaling architectures, which gives more versatility at a lower cost |
| Query Language |	only Sequel | Python, R, Java, Sequel |
| Hardware |	A single server is sufficient | Requires more than one server |
| Value |	Business Intelligence, analysis and reporting | Complex data mining techniques for pattern finding, recommendation, prediction etc. |
| Optimization |	Data can be optimized manually(human powered) | Requires machine learning techniques for data optimization |
| Storage |	Storage within enterprises, local servers etc. | Usually requires distributed storage systems on cloud or in external file systems |
| People |	Data Analysts, Database Administrators and Data Engineers | Data Scientists, Data Analysts, Database Administrators and Data Engineers |
| Security |	Security practices for Small Data include user privileges, data encryption, hashing, etc. | Securing Big Data systems are much more complicated. Best security practices include data encryption, cluster network isolation, strong access control protocols etc. |
| Nomenclature |	Database, Data Warehouse, Data Mart | Data Lake |
| Infrastructure |	Predictable resource allocation, mostly vertically scalable hardware. | More agile infrastructure with horizontally scalable hardware |
| Applications |	Small-scale applications, such as personal or small business data management | Large-scale applications, such as enterprise-level data management, internet of things (IoT), and social media analysis |

## The fours V's of Big Data

### Volume

**Description**: Scale of data An increased amount of storage 

**Attributes**: Petabytes , Exa, Zetta 

**Drivers**:  Increase in Data Sources

### Velocity

**Description**: Data is produced extremely fast and needs to be processed in real time.

**Attributes**: Batch ,Close to real-time, Streaming

**Drivers**: Improved Connectivity and hardware

### Variety

**Description**:  Data comes in all kinds of formats – from structured like SQL databases to unstructured like text content, images or videos.

**Attributes**: Structure, complexity and origin 

**Drivers**:  Mobile technologies, Scalable infrastructure

### Veracity

**Description**: With so much variety, quality and accuracy are variable.

**Attributes**: Consistency and completeness  Integrity

**Drivers**: Cost and traceability Robust ingestion

## We can add an important 5th V !!!

### Vallue

Organizations need to turn their raw data into value.

## Big Data Life Cycle

We can summarize the big data lifecycle consists of five phases.

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/BigDataLifeCycle.png" width="800" height="400">

## Impact of Big Data on People

Big Data has a significant impact on people's lives, both positive and negative. Here are some examples:

Positives:

Improved Decision Making: Big Data helps businesses and governments make more informed decisions. This can lead to better products and services, as well as more effective public policies that, in turn, benefit people.

Health Innovations: Big Data analytics are being used to identify health trends, improve treatments, and even predict disease outbreaks. This can lead to better health and longevity for many people.

Personalization: Big Data can be used to personalize experiences, from product recommendations to personalized learning content.

Negatives:

Privacy: Big Data collection often involves the collection of personal information, which can lead to significant privacy concerns. This can affect personal autonomy and freedom.

Inequality: The use of Big Data can lead to greater inequality. For example, it can be used to target ads in ways that perpetuate discrimination.

Security: Collecting and storing large amounts of data creates opportunities for security breaches, which can have harmful impacts on people.

Information overload: With the increasing use of Big Data, there is a risk of information overload, where the large amount of data available makes it difficult for people to process and make decisions.

## Big Data Tools

### Data Technology

**Description**: Capture, process, and share data at any scale and in any format Work with structured and unstructured data Leverage high-performance, parallel processing of Big Data.

**Common Tools**: Apache Hadoop, Apache HDFS, Apache Spark, Cloudera and Databricks 

### Analytics and Visualization

**Description**: Examines large amounts of data Analyzed data is visualized using graphs, charts and maps.

**Common Tools**: Tableau, Palantir, SAS, Pentaho, and Teradata

### Business Intelligence

**Description**: Offer a range of tools that provide a quick and easy way to transform data into actionable 
insights. Such insights inform an organization strategic and tactical business decisions.

**Common Tools**: Cognos, Oracle, PowerBI, Business Objects, and Hyperion

### Cloud Providers

**Description**: Offer fundamental infrastructure and support with shared computing resources including computing, storage, networking, and analytical software.

**Common Tools**: AWS, IBM, GCP, and Oracle

### NOSQL Databases

**Description**: NoSQL Databases are best for Big Data processing: 

• Store and process vast amounts of data at scale 

• Store information in JSON documents rather than relational tables

• NoSQL databases types include pure document databases, key-value stores, wide column databases, and graph atabases.

**Common Tools**: MongoDB, CouchDB, Cassandra, and Redis

### Programing Tools

**Description**: Perform large-scale analytical tasks and operationalize Big Data. 
Provide all necessary functions for the Big Data Life Cycle.

**Common Tools**: R, Python, SQL, Scala, and Julia

## Open source and Big Data

Open Source Software refers to a type of software whose source code is released under a license that allows users to study, change and improve the software. This code is available for the public to view, copy, modify and distribute.

The open source model is used to Big data because:
- Large and complex projects
- Projects meet the needs of many organizations
- The open source model emerged as the best development strategy

## Big Data Use Cases

Examples:
- An Airline can extract, store, process and analyze passenger travel data in order to offer routes with greater probability of sale.
- A supermarket chain can extract, store, process and analyze purchasing data in order to detect patterns and organize the products in order to increase sales.
- A Hotel Chain can extract, store, process and analyze customer feedback data on social networks to customize your services, increase sales and reduce costs.
- A Hospital Network can extract, store, process and analyze data from medical exams In order to customize and oDmize the care of patients.

## Parallet Processing Scaling and Data Parallelism

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/LinearParallelProcessing.jpg" width="800" height="400">

What is a Computer Cluster?
- A server is a computer, usually with high capacity computational, which “serves” (provides) storage services, applications or databases.
- A server has scalability vertical, that is, there is a limit to how far we managed to include more space in
disk, more processors and more RAM memory. 
- A cluster of computers is a set of servers with the same purpose of providing a type of service such as storage or Data processing. 
- A cluster has scalability horizontal, that is, if we want increase computing power we add more machines to the cluster (in addition to the verNcal scalability of each individual machine in the cluster).
- Computer clusters are increasingly most used in Big Data, which allows storage and parallel processing through multiple machines (multiple servers).

What is Parallel Storage?

- Parallel storage consists of distribute data storage across from several servers (computers), which
makes it possible to significantly increase the storage capacity using low-cost hardware.
- Among some options, Apache Hadoop HDFS (Hadoop Distributed File System) has proven to be the ideal solution to manage storage distributed across a cluster of computers. HDFS is the software responsible for managing the cluster of computers defining how the files will be distributed across the cluster. With HDFS we can build a Data Lake that runs on a cluster of computers and allows storage of large volumes of data
with commodity (low-cost) hardware. This allowed Big Data to be used in Large scale!

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/StorageProcessingArchitectureParallel.png" width="800" height="400">

- In parallel processing, the goal is to divide a task into several subtasks and run them in parallel. Apache Hadoop MapReduce and Apache Spark are two frameworks for this purpose.
- When using a processing framework parallel, the subtasks are taken to the processor of the cluster machine where the data is stored, thus increasing the large processing speed data volumes.

- HDFS is a service running on all cluster machines, being a NameNode to manage the cluster and DataNodes who do the storage work proper.
- MapReduce is also a service running on all machines in the cluster, being a Job Tracker to manage the processing and the Task Trackers that do processing work.
- Job Tracker queries the NameNode in order to to know the location of data blocks on cluster machines.
- Task Trackers communicate with DataNodes to get the disk data, run the processing and then return the result to Job Tracker. 
- This architecture allows storing and process large amounts of data and thus extracting value from Big Data through the data analysis.

## Key Concepts in Big Data

### Cap Theorem

- CAP theorem explains important characteristics for distributed systems. The fundamental idea of the CAP theorem is that in a distributed system, there are three important characteristics, i.e., Consistency, Availability, and Partition Tolerance (CAP). CAP theorem states that in case of a partition (or network failure) both consistency and availability cannot be offered together. That is, when network failure occurs and the network is partitioned, a distributed system can either offer consistency or availability but not both. Note that when there is no network failure, a distributed system can offer both availability and consistency together.

### ACID VS BASE 

ACID PROPERTIES
- **Atomicity (A)** Each transaction should be considered as an atomic unit where either all the operations of a transaction are executed or none are executed. 
- **Consistency (C)** The database will remain in a consistent state after a successful transaction. If a user tries to update a column of a table of type float with a value of type varchar, the update is rejected by the database as it violates consistency 
- **Isolation (I)** Prevents the conflict between concurrent transactions, where multiple users access the same data, and ensures that the data updated by one user is not overwritten by another user. When two users are attempting to update a record, they should be able to work in isolation without the intervention of each other, that is, one transaction should not affect the existence of another transaction. 
- **Durability (D)** Ensures the database will be durable enough to retain all the updates even if the system crashes, that is, once transaction is completed successfully, it becomes permanent. If a transaction attempts to update data in a database and completes successfully, then the database will have the modified data. On the other hand, if a transaction is committed, but the system crashes before the data is written to the disk, then the data will be updated when the system is brought back into action again.

BASE PROPERTIES
- **Basically available** A database is said to be basically available if the system is always available despite a network failure. 
- **Soft state** Soft state means database nodes may be inconsistent when a read operation is performed. For example, if a user updates a record in node A before updating node B, which contains a copy of the data in node A, and if a user requests to read the data in node B, the database is now said to be in the soft state, and the user receives only stale data. 
- **Eventual consistency** The state that follows the soft state is eventual consistency. The database is said to have attained consistency once the changes in the data are updated on all nodes. Eventual consistency states that a read operation performed by a user immediately followed by a write operation may return inconsistent data. For example, if a user updates a record in node A, and another user requests to read the same record from node B before the record gets updated, resulting data will be inconsistent; however, after consistency is eventually attained, the user gets the correct value.

## Deep Learning with Big Data

- Deep Learning, is a sub-area of ​​Machine Learning, which employs algorithms to process data and mimic the processing done by the human brain. Deep Learning uses layers of mathematic neurons to process data, understand human speech and visually recognize objects. Information is passed through each layer, with the output of the previous layer providing input to the next layer. The first layer in a network is called the input layer, while the last one is called the output layer. All layers between the two are referred to as hidden layers. Each layer is typically a simple, uniform algorithm containing one type of activation function.

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/DeepLearning.png" width="800" height="400">
 
A Deep Neural Network (DNN) is distinguished from an NN with respect to the number of hidden layers such that a DNN could have many hidden layers.
At the input layer, the number of nodes is equivalent to the number of input features. Similarly, the number of output nodes is equivalent to the number of 
classes. Number of nodes in the hidden layer could vary between the nodes at the input layer and the output layer.

In big data analytics, there are many features or variables that differentiate an 
object from another object. For instance, a fruit can be differentiated with its 
color, shape, and size. Similarly, a car may have unique features of color, 
shape, position, brightness, and size. The factors that capture the sense of 
variability in data are called as factors of variation. These factors are useful in 
classifying a particular object. When looking for an object, we tend to utilize the 
related features and tend to ignore the features that are irrelevant. 
An NN (or a DNN) could be helpful in this context. It can be used to discover 
the representation and map the output from the representation. The class 
of machine learning which discovers the representation of features and maps 
them to an output is known as representation learning. A DNN simplifies the 
complex problem of identifying representations by introducing layers, each of 
which is responsible to extract meaningful information. 
This powerful usage of DNN in representation learning has assisted in finding 
intelligent solutions for many big data problems such as natural language 
processing, image recognition, handwriting recognition, and speech 
recognition. 

**Feedforward neural networks (FF or FFNN) and perceptrons (P)** are very straightforward, they feed information from front to back (input and output respectively). Neural networks are often described as having layers, where each layer consists of parallel input, hidden, or output cells. A layer alone never has connections and in general two adjacent layers are fully connected (each neuron forms a layer for each neuron for another layer). The simplest and most practical network has two input cells and one output cell, which can be used to model logic gates. Typically, FFNNs are trained via backpropagation, providing the network with paired “what in” and “what do we want out” datasets to the network. This is called supervised learning, as opposed to unsupervised learning, where we just give input and let the network fill in the blanks. The error being propagated back is usually some variation of the difference between the input and the output (such as MSE or just the linear difference). Given that the network has enough hidden neurons, it can theoretically always model the relationship between input and output. Practically their use is much more limited, but they are popularly combined with other networks to form new networks.

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/FeedForwardNeuralNetwork.png" width="800" height="400">


**Recurrent neural networks (RNN)** are a powerful set of artificial neural network algorithms especially useful for processing sequential data such as sound, time series data or natural language.
Recurrent networks differ from feed forward networks in that they include a feedback loop, whereby the output of step n-1 is fed back into the network to affect the outcome of step n, and so on for each subsequent step.
Recurrent networks produce dynamic models – that is, models that change over time – in ways that produce accurate classifications dependent on the context of the examples that are exposed. RNN has is used in a number of big data applications including machine translation, speech recognition, and text summarization

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/RecurrentNeuralNetworks.png" width="800" height="400">

**Convolutional Neural Networks (ConvNets or CNNs)** are deep artificial neural networks that can be used to classify images, group them by similarity (picture search) and perform object recognition within scenes.
Convolutional networks ingest and process images like tensors, and tensors are arrays of numbers with multiple dimensions.
Convolutional networks perceive images as volumes; that is, three-dimensional objects rather than flat structures to be measured only by width and height. That's because digital nuclei images have a red-green-blue (RGB) mix, mixing these three nuclei together to produce the spectrum of nuclei that humans perceive. A convolutional network is imaged as three separate strata of cores stacked on top of each other.
Thus, a convolutional network receives an image as a rectangular box whose width and are measured by the number of pixels along those dimensions and whose depth is three layers deep, one for each letter in RGB. These depth layers are referred to as channels.
As images move through a convolutional network, we describe them in terms of input and output volumes, expressing them mathematically as multidimensional matrices like this: 30x30x3. From layer to layer, its dimensions change as they traverse the convolutional neural network until it generates a series of probabilities in the output layer, one probability for each possible output class. The one with the highest probability will be a class defined for the input image, a bird for example.
CNNs are extremely useful for applications such as
Optical Character Recognition (OCR), image recognition,
and facial recognition.

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/ConvolutionalNeuralNetworks.png" width="800" height="400">

## Hadoop is an open source framework for big data

Hadoop is an open-source software framework that enables distributed processing of large datasets across clusters of computers using simple programming models. It is designed to scale from a single server to thousands of machines, each offering local storage and computing power. In other words, Hadoop is a framework that allows manipulation and analysis of big data. It is based on the concept of MapReduce, where data is divided into smaller blocks for faster and more efficient processing.

Hadoop is made up of four main modules:

**Hadoop Common**: These are the common libraries and utilities that support the other Hadoop modules.

**Hadoop Distributed File System (HDFS)**: This is the distributed storage system used by Hadoop. It is capable of storing large amounts of data (at the scale of petabytes) and providing high-speed access to that data.

***HDFS concepts***

- Block
The minimum amount of data that can be read or written
- Node
System responsible to store and process data
- Rack
The collection of about 40 to 50 data nodes using the same network switch. Data node racks that are closer to each other improves cluster performance by reducing network traffic.
- Replication
Copies of data blocks are created for back up purposes Write once read many operations

***HIVE***

- Hive is a data warehouse software within Hadoop that is designed to read, write, and manage large and tabular-type datasets and data analysis 
- It is scalable, fast, and easy to use
- Hive Query Language is inspired by SQL making it easier for users to grasp concepts
- It supports data cleansing and filtering depending on user requirements
  
**Hadoop YARN**: (Yet Another Resource Negotiator) is the resource manager and task scheduler for Hadoop. It manages system resources and schedules tasks for different applications.

**Hadoop MapReduce**: This is the programming model for processing large datasets. It divides processing tasks into small chunks of work, which can be run or rerun on any node in the cluster.

***How Map-Reduce works***

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/MapReduceWorks.png" width="800" height="400">

Hadoop is used in many areas, including business analytics, scientific research, and image processing, among others. One of the great advantages of Hadoop is its ability to efficiently process, analyze and store large volumes of data.

## Apache Spark

Apache Spark is an open source cluster computing framework that provides an interface for programming complete clusters with performance optimization and support for large-scale data streams. It was designed to be fast and general purpose and excels at processing large datasets.

Spark is often compared to Hadoop, but in reality, the two are best described as complementary rather than competing technologies. While both are useful for processing large datasets, Spark can perform real-time tasks with great speed, whereas Hadoop more traditionally does batch work.

Spark includes four main components:

Spark Core: It is the fundamental execution engine that supports general I/O tasks and distributed scheduling.

Spark SQL: It is an interface for working with structured and semi-structured data.

Spark Streaming: Allows the processing of data streams in real time.

MLlib (Machine Learning Library): It is a library of machine learning algorithms for training models and making large-scale predictions.

GraphX: It is a library for manipulating graphs (eg social networks) and performing calculations in parallel.

***Parallel computing VS Distributed computing***

- Parallel computing processors access shared memory 
- Distributed computing processors usually have their own private or distributed memory
-- Scalability and modular growth
-- Fault Tolerance and redundacy

***Apache Spark and Map Reduce compared***

<img src="https://raw.githubusercontent.com/E-man85/Big-Data/main/04-images/ApacheSparkMapReduceCompared.png" width="800" height="400">

## Spark Structured Streaming

Is continuously generated Often originates from more than one source Is unavailable as complete data set
Requires incremental processing.

**APACHE SPARK STRUCTURED STREAMING** 
- Uses Spark SQL
- Uses the same DataFrame and Dataset APIs
- Processes data in Micro- bactches or continuously
- Optimizes queries using Spark SQL

**COMMON STRUCTURED STREAMING TERMS**
- Soure: the data origination location
- Sink: the location of the output
- Event time: the record creation time
- End to end latency: The measurement of the time needed for the data to go from sources to sink
- Watermarking: manages late arriving data

**STREAMING DATA SOURCES**
- Files – streams files from a directory
- Kafka – streams from apache Kafka
- IP Sockets and Rate sources – used for testing only

**STREAMING DATA OPERATIONS**

Apache Spark Structured Streaming:

- Performs Standard SQL operations including, select, projection and aggregation
- Enables window operations over event time – sliding windows with aggregations
- Supports join operations – joins with static DataFrames or other streams

**OUTPUT MODES**

Output modes specify how data is written to the sink

- Append – only new rows added
- Complete – entire results table
- Update – only updated rows

**STREAMING DATA SINKS**
- Files Outputs files to a directory
- Kafka outputs to Kafka topics
- Foreach and ForreachBatch Applies a function to each record batch
- Console and Memory Needed for Debugging

**MONITORING AND CHECKPOINTING**
- Monitor your data via Spark external listeners, which work both trough externa frameworks or programmatically 
- Checkpointing recovers query progress on failure (set checkpoint location on HDFS)

## GraphFrames on Apache Spark

**GRAPH THEORY**
Graph Theory is the mathematical study of modelling parwise relationships between objects

Graphs consist of: 
- Vertices
- Edges that connect one vertex to another

Graphs can either be directed or undirected

**Directed graphs** Contain edges with a single direction between two vertices

Examples: manufacturing optimization, project scheduling, train and airline route analysis, traffic recommendations and others

**Undirected graphs** Contain edges with no defined directions

Examples: social relationship analysis, marketing analysis, geonomics analysis, knowledge repositories, and others

**GRAPHFRAMES – WHAT IS IT?**
- An Apache Spark extension for graph processing Based on Spark DataFrames 
- Runs queries on graphs of vertices and edges and represents data
- Contains built-in algorithms
- Exists as a separate downloadable package

**USING GRAPHFRAMES**
- Runs SQL queries on vertex and edge DataFrames
- Requieres that you set a directory for checkpoints
- Performs Motif finding, which searches the graph for structurual patterns

**SUPPORTED GRAPH ALGORITHMS**
- Breath first search (BFS)
- Connected components (strongly connected components)
- Label Propagation Algorithm
- Page Rank
- Shortest path
- Triangle count

**PAGE RANK GRAPH ALGORITHMS**
Developed by Google to measure importance and rank webpages for their search engine

**SUPPORTED TYPES OF DATA**
- Ideal for modelling data with connecting relationships
- Computes relationship strength and direction

## ETL Workloads on Apache Spark

**ETL – WHAT IS IT**
- ETL describes the process of moving data froma source to another destination that has a different data format or structure
- Extract obtains data from a source
- Transform the data in the needed output format
- Load the data into a database, data warehouse or other storage

**ETL – WITH APACHE SPARK**
- Apache Spark offers the following ETL advantages 
- Provides a well supported big data ecosystem
- Can easily load and save popular big data sources
- Scales easily to handle large workloads

**EXTRACTING DATA**
Extracts data from one or more different sources

Spark supports HDFS & the following data sources:
- Parquet
- Apache ORC
- Hive
- JDBC

**TRANSFORMING THE DATA**
- Cleans the data
- Transforms data format to make the data more accessible for analysis
- Joins DataFrames
- Groups and aggregates data
- Uses Spark SQL operations to select others

**LOADING DATA**
- Loads data into data warehouse, database or other data sink
- Uses available Spark data sources

## Spark ML Funfamentals

**WHAT IS MACHINE LEARNING**
- Applies algorithms that automatically learn features from data
- Are not explicitly programmed to do a task
- Learn from data/experience and improve with more data
- Apply statistical tools that enables AI
- Uses models trained with data and performs predictions with new data

**MACHINE LEARNING AND SPARK**
- Spark ML Library is also commonly called Mllib
- Applies practical machine learning algorithms an is scalable
- Performs machine learning operations using DataFram-base API’s

**SPARK MLIB DATA SOURCES**

Supports standard data sources including:
- Parquet
- CSV
- JSON
- JDBC
- Supports machine learning specific data sources
- Has special libraires to support images and LIBSVM data types
- Supports both feature vector and label column data
- Images are a common data source and create a DataFrame and an image schema
- LBSVM loads the libsvm data files and creates a DataFrame with 2 columns including the feature vector 
and label.

**SPARK INBUILT UTILITIES**
- Linear Algebra: used for basic linear algebra data holders such as matrices and vectors
- Statistics: used for statistics operations such as correlation, hypothesus testing, summarizing etc
- Feature: powerful toolbox to convert raw data into useful features for ML model fitting

**ML PIPELINES**
- Exposes a single Spark ML pipeline API 
- Combines multiple algorthms into a single workflow or pipeline using transformers and estimators as building blocks
- Transformers can transform one DataFrame using the transform function 

For example: A models that converts feature data into predictions

Estimators encapsulate the algorithm and model learning on the feature using the fit function


## Resources

[geeksforgeeks.org](https://www.geeksforgeeks.org/difference-between-small-data-and-big-data/)

[itrelease.com](https://www.itrelease.com/2017/11/difference-serial-parallel-processing/)

[datascienceacademy](https://www.datascienceacademy.com.br/)

[deeplearningbook](https://www.deeplearningbook.com.br/)

[asimovinstitute](https://www.asimovinstitute.org/overview-neural-network-zoo/)
