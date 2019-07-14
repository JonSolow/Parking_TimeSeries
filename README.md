# Module 4 -  Final Project Specifications

## Introduction

In this lesson, we'll review all the guidelines and specifications for the final project for Module 4. 

## Objectives

* Understand all required aspects of the Final Project for Module 4
* Understand all required deliverables
* Understand what constitutes a successful project

### Final Project Summary

Another module down--you're absolutely crushing it! For this project, you'll get to flex your **_Time-Series_** muscles!
<br>
<br>
<center>
<img src='images/time.webp'>
<a href='https://media.giphy.com/media/l0MYOUI5XfRk4LLWM/giphy.gif'>Via Giphy</a></center>
<br>
<br>
For this module's final project, we're going to put your newfound **_Time Series Analysis_** skills to the test. You will be forecasting real estate prices of various zipcodes using data from [Zillow](https://www.zillow.com/research/data/). However, this won't be as straightforward as just running a time-series analysis--you're going to have to make some data-driven decisions and think critically along the way!

### The Project

For this project, you will be acting as a consultant for a fictional real-estate investment firm. The firm has asked you what seems like a simple question:

> what are the top 5 best zipcodes for us to invest in? (e.g. the hottest)

This may seem like a simple question at first glance, but there's more than a little ambiguity here that you'll have to think through in order to provide a solid recommendation. Should your recommendation be focused on profit margins only? What about risk? What sort of time horizon are you predicting against?  Your recommendation will need to detail your rationale and answer any sort of lingering questions like these in order to demonstrate how you define "best". 

As mentioned previously, the data you'll be working with comes from the [Zillow Research Page](https://www.zillow.com/research/data/). However, there are many options on that page, and making sure you have exactly what you need can be a bit confusing. For simplicity's sake, we have already provided the dataset for you in this repo--you will find it in the file `zillow_data.csv`.

In terms of the analysis, please make sure to include:

	- Baseline model
	- Staitonarity (e.g. trends, seasonal patterns, cyclical patterns, differencing, Dickey Fuller Test)  
	- ACF
	- PACF
	- At least AR and MA models
	- Evaluation metric  

## The Deliverables

The goal of this project is to have you complete a very common real-world task in regard to Time-Series Modeling. However, real world problems often come with a significant degree of ambiguity, which requires you to use your knowledge of statistics and data science to think critically about and answer. While the main task in this project is Time-Series Modeling, that isn't the overall goal--it is important to understand that Time-Series Modeling is a tool in your toolbox, and the forecasts it provides you are what you'll use to answer important questions. 

In short, to pass this project, demonstrating the quality and thoughtfulness of your overall recommendation is at least as important as successfully building a Time-Series model!

In order to successfully complete this project, you must have:

* A well-documented **_Jupyter Notebook_** containing any code you've written for this project 
* An **_'Executive Summary' PowerPoint Presentation_** that explains your rationale and methodology for determining the best zipcodes for investment.
* A shared repo with at least 10 commits 


### Jupyter Notebook Must-Haves

For this project, your jupyter notebook should meet the following specifications:

#### Organization/Code Cleanliness

The notebook should be well organized, easy to follow, and code is modularized and commented where appropriate.

* Level Up: The notebook contains well-formatted, professional looking markdown cells explaining any substantial code. All functions have docstrings that act as professional-quality documentation. 
* The notebook is written to technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings. 
* Data visualizations you create should be clearly labeled and contextualized--that is, they fit with the surrounding code or problems you're trying to solve. No dropping data visualizations randomly around your notebook without any context!

#### Findings

Your notebook should briefly mention the metrics you have defined as "best", so that any readers understand what technical metrics you are trying to optimize for (for instance, risk vs profitability, ROI yield, etc.). You do **not** need to explain or defend your your choices in the notebook--the executive summary presentation are both better suited to that sort of content. However, the notebook should provide enough context about your definition for "best investment" so that they understand what the code you are writing is trying to solve. 

#### Visualizations

Time-Series Analysis is an area of data science that lends itself well to intuitive data visualizations. Whereas we may not be able to visualize the best choice in a classification or clustering problem with a high-dimensional dataset, that isn't an issue with Time Series data. As such, **_any findings worth mentioning in this problem are probably also worth visualizing_**. Your notebook should make use of data visualizations as appropriate to make your findings obvious to any readers. 

Also, remember that if a visualization is worth creating, then it's also worth taking the extra few minutes to make sure that it is easily understandable and well-formatted. When creating visualizations, make sure that they have:

* A title
* Clearly labeled X and Y axes, with appropriate scale for each
* A legend, when necessary
* No overlapping text that makes it hard to read
* An intelligent use of color--multiple lines should have different colors and/or symbols to make them easily differentiable to the eye
* An appropriate amount of information--avoid creating graphs that are "too busy"--for instance, don't create a line graph with 25 different lines on it

<center><img src='images/bad-graph-1.png' height=100% width=100%>
There's just too much going on in this graph for it to be readable--don't make the same mistake! (<a href='http://genywealth.com/wp-content/uploads/2010/03/line-graph.php_.png'>Source</a>)</center>

### Executive Summary Must-Haves

Your presentation should:

Contain between 5-10 professional quality slides detailing:

* A high-level overview of your methodology and findings, including the 5 zipcodes you recommend investing in
* A brief explanation of what metrics you defined as "best" in order complete this project

As always, this prresentation should also:

* Take no more than 5 minutes to present 
* Avoid technical jargon and explain results in a clear, actionable way for non-technical audiences.