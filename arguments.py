import argparse

def params():
    parser = argparse.ArgumentParser()


    # paths and models
    parser.add_argument("--model-to-train", default="meta-llama/Meta-Llama-3-8B-Instruct", type=str,\
                help="model to train (get from HuggingFace)")
    parser.add_argument("--model-name", default="Llama-3-8B-Instruct-spider-4", type=str,\
                help="model name to save the model as")
                # with ppo, the model is saved as [model_name]-ppo, and with dpo, saved as [model_name]-dpo
    parser.add_argument("--datasets-path", default="../spider_datasets", type=str,\
                help="path to save the datasets")
    parser.add_argument("--databases-path", default="../spider_databases", type=str,\
                help="path to save the databases")
    parser.add_argument("--dataset-on-hub", default="VictorDCh/spider-clean-text-to-sql-3", type=str,\
                help="dataset on HuggingFace hub")
                # VictorDCh/spider-clean-text-to-sql or -2 or -3 or -4 (-3 has been noticed to be the best)
    parser.add_argument("--cot-model-id", default="meta-llama/Meta-Llama-3-8B-Instruct", type=str,\
                help="COT model id")
    

    # dataset options
    parser.add_argument("--prepare-datasets", default=True, type=bool,\
                help="prepare the datasets")
                # you only need to set True once in a new environment, afterwards, set to False, except if changing rag config
                # as of the latest version of the spider dataset used, setting this to True doesn't change much in terms of computing time
    parser.add_argument("--prepare-dpo-dataset", default=False, type=bool,\
                help="prepare the DPO dataset")
                # uses the SFT model to generate the dataset (needs a preferred/rejected dataset)
    parser.add_argument("--use-gretel", default=True, type=bool,\
                help="use the dataset from Gretel AI to have a more diverse training dataset (SFT only)")
                # gretelai/synthetic_text_to_sql on HF
    parser.add_argument("--nb-gretel-samples", default=7000, type=int,\
                help="number of samples to use from the Gretel dataset")
                # the gretel dataset contains 100k samples, so you can use less if you want for training


    # HF login tokens and username
    parser.add_argument("--hf-token-read", default="read_token_here", type=str,\
                help="HuggingFace token for reading")
    parser.add_argument("--hf-token-write", default="write_token_here", type=str,\
                help="HuggingFace token for writing")
    parser.add_argument("--hf-username", default="username_here", type=str,\
                help="HuggingFace username")


    # dpo dataset creation
    parser.add_argument("--create-dpo-dataset", default=False, type=bool,\
                help="create the DPO dataset")
                # quite long to do as it requires a model to generate the dataset
                # already done, don't set to True unless you want to change the dataset
    

    # training and testing
    parser.add_argument("--train", default=True, type=bool,\
                help="train the model")
    parser.add_argument("--train-dpo", default=False, type=bool,\
                help="train the model with DPO")
                # supposedly more efficient but less performant than PPO
    parser.add_argument("--train-ppo", default=False, type=bool,\
                help="train the model with PPO")
    
    parser.add_argument("--test", default=True, type=bool,\
                help="test the model")
                # needs to be set to True to test the model for dpo or ppo
    parser.add_argument("--test-dpo", default=False, type=bool,\
                help="test the model with DPO")
                # not yet implemented
    parser.add_argument("--test-ppo", default=False, type=bool,\
                help="test the model with PPO")


    # other booleans
    parser.add_argument("--quantize", default=True, type=bool,\
                help="quantize the model in 4 bits when running inference")
    parser.add_argument("--eval", default=False, type=bool,\
                help="evaluate the model during training")
                # set to False for big models, such as a MoE, as there will not be enough GPU memory for it
                

    # advanced inference methods
    parser.add_argument("--use-rag", default=False, type=bool,\
                help="use RAG model")
    parser.add_argument("--use-cot-rectification", default=False, type=bool,\
                help="use COT in evaluation to rectify the model's output")
                # chain of thought in evaluation (evaluation takes longer (2x))
    parser.add_argument("--use-schema-linking", default=False, type=bool,\
                help="use schema linking in evaluation through CoT")
                # schema cleaning in evaluation through CoT (will ask the model to remove unnecessary tables and columns)
    

    # chat format
    parser.add_argument("--conv-roles", default="sys-user-assistant", type=str,\
                help="conversation roles")
                # can be sys-user-assistant or user-assistant
                # usually user-assistant is used only for mistralai instruct models
    parser.add_argument("--rag-input-format", default="question", type=str,\
                help="RAG input format")
                # can be schema_and_question or question (more like what is accounted for in the similarity search)


    # num tests and train
    parser.add_argument("--num-tests", default=-1, type=int,\
                help="number of tests to run")
                # -1 for all
    parser.add_argument("--num-train", default=-1, type=int,\
                help="number of training samples")
                # -1 for all
    parser.add_argument("--num-ppo", default=-1, type=int,\
                help="number of samples for ppo training")
                # -1 for all
    parser.add_argument("--epochs", default=4, type=int,\
                help="number of epochs to train the model")
    parser.add_argument("--epochs-dpo", default=4, type=int,\
                help="number of epochs to train the model with DPO")
    parser.add_argument("--nb-rag-samples", default=3, type=int,\
                help="number of few shot samples to use")


    # seed
    parser.add_argument("--seed", default=42, type=int,\
                help="seed for reproducibility")


    args = parser.parse_args()
    return args
