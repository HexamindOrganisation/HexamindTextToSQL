# Spider text-to-SQL generation



## Authors
Victor Dubus-Chanson, Hexamind

## Description
The goal of this project is to generate SQL queries from natural language questions using a large language model (LLM) with varying tweaks, in its architecture and through fine-tuning and other methods. The project was essentially made to be tested on the Spider benchmark, which is a large dataset of natural language questions and their corresponding SQL queries.
[Here is a medium article](https://medium.com/@vdubuschanson/training-small-open-source-llm-for-text-to-sql-generation-7586d6084346) to better understand the goal and what was achieved in this project.

## Table of contents
- [Authors](#authors)
- [Description](#description)
- [Table of contents](#table-of-contents)
- [Disclaimer](#disclaimer)
- [Setup](#setup)
- [Warnings](#warnings)
- [Data](#data)
- [Structure](#structure)



### Disclaimer
- Most of this project was designed to work on a Scaleway GPU instance.
- Only two main datasets can be used as is with this project: variations of VictorDCh/spider-clean-text-to-sql-3 and gretelai/synthetic_text_to_sql (although not in the same way, check arguments.py and dataset_prep.py for more information).

### Setup
- If using a Scaleway GPU instance, run this command to work in a container with CUDA, CUDnn, PyTorch and other libraries installed: "docker run --runtime=nvidia -it --rm -p 8888:8888 -p 6006:6006 rg.fr-par.scw.cloud/scw-ai/pytorch:latest /bin/bash". Go to [Scaleway's website](https://www.scaleway.com/en/docs/compute/gpu/reference-content/docker-images/) for more information.
- Other than the dependencies you can get through the above container, some libraries are required. Run the cells in `requirements.ipynb` to install them.

### Warnings
- The HuggingFace tokens in arguments.py might be None, so make sure to change them to the correct values. If the tokens are already set, they are probably not yours and it is a mistake on our part, so please change them to your own tokens.
- One other place where you might need to change the tokens is in the last cell of merger.ipynb, where you will need to change the tokens to your own.
- In requirements.ipynb, you will need to add your email adress and username for git to work properly.

### Data
- You will need to have in the parent directory of this project a part of the spider dataset, which you can download from the Spider website. The part you will need is the folder named "spider" which contains three other folders: "database", "test_database" and "test_data". You only need those three folders in the folder spider. If you do not modify this project, call the "spider" folder "spider_databases".
- The other data you need will be downloaded automatically from Hugging Face. If you want to know how they have been created, look at [this repository](https://github.com/HexamindOrganisation/spider_raw_dataset_prep)

---

### Structure
- "arguments.py" contains the arguments for the different scripts.
- "main.py" is the main script that will run everything that is contained in this repo, except for the merging of models into MoEs or other merges.
- "merger.ipynb" is a notebook that will merge the models into MoEs or other merges.
- "config.yaml" is the configuration file for the creation of merges.
- "requirements.ipynb" is a notebook that will install the required libraries.
- "dataset_prep.py" is a script that will prepare the dataset for the training and inference of the model.
- "dpo_dataset_creation.py" is a script that will create the dataset for the training via DPO of the model.
- "evaluate_model.py" is a script that will evaluate the model on the Spider dataset.
- "rag.py" contains utilities necessary when using few-shot prompting via RAG for the inference of the model.
- "rl_training.py" is a script that will train the model via reinforcement learning (DPO or PPO for now) and upload it to Hugging Face.
- "training.py" is a script that will train the model on the Spider dataset and upload the model to Hugging Face.
- "utils.py" contains utilities necessary for various parts of the project, such as final dataset preparation, metrics calculation, generation checks, etc.

---

### TODO
- [ ] Remove default HF tokens
- [ ] Add more information to the README
- [ ] Add more information to the scripts
- [ ] Remove default git configuration
