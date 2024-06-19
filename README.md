---
title: Leveraging Retrieval-Augmented Generation to Optimize Financial Data Extraction and Analysis
emoji: 🏆
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# RAG-Financial-Insights-through-10-K-Report-Analysis
Financial Insights through AI-Powered 10-K Report Analysis


## Leveraging Retrieval-Augmented Generation to Optimize Financial Data Extraction and Analysis

"Automated Information Extraction and Analysis from 10-K Reports using Retrieval-Augmented Generation (RAG)"
"Enhancing Financial Insights through AI-Powered 10-K Report Analysis"
"Streamlining Investment Recommendations with a RAG-based 10-K Report Processing System"
"Leveraging Retrieval-Augmented Generation to Optimize Financial Data Extraction and Analysis"
"AI-Driven 10-K Report Processing: Improving Financial Analysts' Productivity and Insights"

RAG-investment-recommendation-log

# Problem Statement

Finsights Grey Inc. is an innovative financial technology firm that specializes in providing advanced analytics and insights for investment management and financial planning. The company handles an extensive collection of 10-K reports from various industry players, which contain detailed information about financial performance, risk factors, market trends, and strategic initiatives. Despite the richness of these documents, Finsights Grey's financial analysts struggle with extracting actionable insights efficiently in a short span due to the manual and labor-intensive nature of the analysis. Going through the document to find the exact information needed at the moment takes too long. This bottleneck hampers the company's ability to deliver timely and accurate recommendations to its clients. To overcome these challenges, Finsights Grey Inc. aims to implement a Retrieval-Augmented Generation (RAG) model to automate the extraction, summarization, and analysis of information from the 10-K reports, thereby enhancing the accuracy and speed of their investment insights.

## Objective

As a Gen AI Data Scientist hired by Finsights Grey Inc., the objective is to develop an advanced RAG-based system to streamline the extraction and analysis of key information from 10-K reports. You are asked to deploy a Gradio app on HuggingFace spaces that can RAG 10-k reports and answer the questions of financial analysts swiftly.

The project will involve testing the RAG system on a current business problem. The Financial analysts are asked to research major cloud and AI platforms such as Amazon AWS, Google Cloud, Microsoft Azure, Meta AI, and IBM Watson to determine the most effective platform for this application. The primary goals include improving the efficiency of data extraction. Once the project is deployed, the system will be tested by a financial analyst with the following questions. Accurate text retrieval for these questions will imply the project's success.

## Questions

1. Has the company made any significant acquisitions in the AI space, and how are these acquisitions being integrated into the company's strategy?
2. How much capital has been allocated towards AI research and development?
3. What initiatives has the company implemented to address ethical concerns surrounding AI, such as fairness, accountability, and privacy?
4. How does the company plan to differentiate itself in the AI space relative to competitors?

Each Question must be asked for each of the five companies by adding the company's name to the question. For example,

- Has Google made any significant acquisitions in the AI space, and how are these acquisitions being integrated into the company's strategy?
- How much capital has been allocated towards AI research and development by Google?

## Goals

By successfully developing this project, we aim to:

1. Improve the productivity of financial analysts by providing a competent tool.
2. Provide timely insights to improve client recommendations.
3. Strengthen FinTech Insights Inc.'s competitive edge by delivering more reliable and faster insights to clients.

## Dataset

The `dataset.zip` contains the following five 10k reports in pdf format.

- `google-10-k-2023`
- `aws-10-k-2023`
- `msft-10-k-2023`
- `IBM-10-k-2023`
- `Meta-10-k-2023`

## Best Practices for the Notebook

1. The final notebook should be well-documented, with inline comments explaining the functionality of code and markdown cells containing comments on the observations and insights.
2. The notebook should be run from start to finish sequentially before submission.
3. It is important to remove all warnings and errors before submission.
4. The notebook should be submitted as an HTML file not as a notebook file (.ipynb)


## How to Use This Repository

## Step 1: Clone this respository

[GitHub Repository](https://github.com/mayankchugh-learning/)

## Step 2: Create and activate Conda envirnoment

```bash
conda create -p companiesreportsvenv python -y
```

## Step 3: Install Dependencies

```bash
source activate ./companiesreportsvenv
```
## Step 4: Clone the Project and Run

```bash
pip install -r requirements.txt
```
## Step 4: start the application

```bash
python app.py
```


### Steps to upload files using Hugging Face CLI:
Install huggingface CLI - 
```bash
pip install -U "huggingface_hub[cli]"
```
Upload files - 
```bash
huggingface-cli upload repo_name local_path path_in_repo --token=$hf_token --repo-type=space
```
export hf_token=<KEY>
echo $hf_token
```
Example: 
```bash
huggingface-cli upload username/10kreports ./reports_db ./reports_db/ --token=hf_wPFDRMmQVaflibabflebhtizEFD --repo-type=space

huggingface-cli upload username/mayankchugh-learning ./reports_db ./reports_db/ --token=$hf_token --repo-type=space

```