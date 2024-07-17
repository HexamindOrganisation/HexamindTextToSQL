import sqlite3
from itertools import permutations
from arguments import params
from Levenshtein import distance as levenshtein_distance

args = params()


###################################################################################
### Unused variables and functions to force the model into a specific direction ###
distinct_glossary = ["different", "distinct", "unique", "each", "every", "no duplicates",\
                      "no repeating", "no repeating values", "no repeating rows",\
                      "no repeating entries", "no repeating records", "no repeating data",\
                      "no repeating information", "no repeating values", "no repeating items",\
                      "no repeating elements", "no repeating instances", "no repeating occurrences"]

count_glossary = ["count", "number of", "how many", "how much", "total", "sum",\
                "how often", "how frequently", "occurrences", "instances", "times"]

order_by_glossary = ["order by", "sort", "arrange", "sequence", "sequence by", "sort by"]


# unused
def is_in_gloss(question, glossary):
    '''
    Function to check if the question is asking for distinct values

    :param question: the question to be checked

    :return: True if the question is asking for distinct values, False otherwise
    '''
    question = question.lower()
    for word in glossary:
        if word in question:
            return True

# unused
def force_distinct(question):
    '''
    Function to add an instruction to the question to force distinct statement

    :param question: the question to be modified

    :return: the modified question
    '''
    if is_in_gloss(question, distinct_glossary):
        question = "\nMake sure to return distinct values according to what is asked in the question, with a DISTINCT statement.\n" + question
    return question

# unused
def force_count(question):
    '''
    Function to add an instruction to the question to force count statement

    :param question: the question to be modified

    :return: the modified question
    '''
    if is_in_gloss(question, count_glossary):
        question = "\nMake sure to return a count of certain values according to what is asked in the question, with a COUNT statement.\n" + question
    return question

# unused
def force_order_by(question):
    '''
    Function to add an instruction to the question to force order by or asc or desc statement

    :param question: the question to be modified

    :return: the modified question
    '''
    if is_in_gloss(question, order_by_glossary):
        question = "\nMake sure to order the values according to what is asked in the question, with an ORDER BY or ASC or DESC statement.\n" + question
    return question
### --------------------------------------------------------------------------- ###
###################################################################################



#########################################################################
### Utility functions for checking and changing tables names in query ###
def get_table_names(schema):
    '''
    Function to return the table names from the schema

    :param schema: the schema to get the table names from

    :return: the table names
    '''
    table_names = []
    schema = schema.lower()
    schema = schema.split('\n')
    for line in schema:
        if 'create table if not exists' in line:
            table_name = line.split()[5]
            table_name = table_name.replace('"', '')
            table_name = table_name.replace("'", '')
            table_name = table_name.replace(',', '')
            table_name = table_name.replace('`', '')
            table_name = table_name.replace(';', '')
            table_name = table_name.replace('(', '')
            table_name = table_name.replace(')', '')
            table_names.append(table_name)
        elif 'create table' in line:
            table_name = line.split()[2]
            table_name = table_name.replace('"', '')
            table_name = table_name.replace("'", '')
            table_name = table_name.replace(',', '')
            table_name = table_name.replace('`', '')
            table_name = table_name.replace(';', '')
            table_name = table_name.replace('(', '')
            table_name = table_name.replace(')', '')
            table_names.append(table_name)
    return table_names


def similarity_search(wrong_table, tables):
    '''
    Function to return the most similar table name to the wrong table name
    This function uses the levenstein distance to calculate the similarity

    :param wrong_table: the wrong table name
    :param tables: the correct table names

    :return: the most similar table name
    '''
    min_distance = -1
    similar_table = ''
    for table in tables:
        distance = levenshtein_distance(wrong_table, table)
        if distance < min_distance or min_distance == -1:
            min_distance = distance
            similar_table = table
    return similar_table

def get_table_names_from_query(query):
    '''
    Function to return the table names from the query

    :param query: the query to get the table names from

    :return: the table names
    '''
    table_names = []
    query = query.lower()
    query = query.split(' ')
    for word in query:
        if word in ['from', 'join']:
            table_name = query[query.index(word) + 1]
            # don't count the word if it begins with a (
            if len(table_name) > 2:
                if not (table_name[0] == '(' or table_name[1] == '(' or table_name[2] == '('):   
                    # if the table name has a comma, remove it
                    table_name = table_name.replace(',', '')
                    # if the table name has a quote or double quote, remove it
                    table_name = table_name.replace("'", '')
                    table_name = table_name.replace('"', '')
                    table_name = table_name.replace('`', '')
                    table_name = table_name.replace(';', '')
                    table_name = table_name.replace('(', '')
                    table_name = table_name.replace(')', '')
                    table_names.append(table_name)
            else:
                table_names.append(table_name)
    return table_names

def check_table_names(query, schema):
    '''
    Function to check if the table names in the query are correct
    If the table names are incorrect, the function returns the most similar table names

    :param query: the query to check the table names
    :param schema: the schema to get the correct table names

    :return: True if the table names are correct, False otherwise
    '''
    table_names = get_table_names(schema)
    query_table_names = get_table_names_from_query(query)
    for query_table_name in query_table_names:
        if query_table_name not in table_names:
            return False
    return True

def correct_query(query, schema):
    '''
    Function to correct the query if the table names are incorrect
    The function returns the corrected query

    :param query: the query to be corrected
    :param schema: the schema to get the correct table names

    :return: the corrected query
    '''
    table_names = get_table_names(schema)
    query_table_names = get_table_names_from_query(query)
    #print("query table names: ", query_table_names)
    #print("true table names: ", table_names)
    for query_table_name in query_table_names:
        if query_table_name not in table_names:
            similar_table = similarity_search(query_table_name, table_names)
            similar_table = similar_table.replace('"', '')
            low_query = query.lower()
            split_low_query = low_query.split()
            split_query = query.split()
            try:
                split_query[split_low_query.index(query_table_name)] = similar_table
            except:
                print("Minimal error when looking for the table name, seems it was already replaced. Continuing.")
            query = ' '.join(split_query)
    return query
### ----------------------------------------------------------------- ###
#########################################################################


################################################
### Metrics and utilities to compute metrics ###
def exact_match_accuracy(y_true, y_pred):
    '''
    Calculate the exact match accuracy of the model

    :param y_true: true query
    :param y_pred: predicted query

    :return: exact match accuracy
    '''

    """
    # lower everything
    y_true = y_true.lower()
    y_pred = y_pred.lower()

    # remove all spaces or tabs or newlines or returns
    y_true = y_true.replace(' ', '')
    y_pred = y_pred.replace(' ', '')
    y_true = y_true.replace('\t', '')
    y_pred = y_pred.replace('\t', '')
    y_true = y_true.replace('\n', '')
    y_pred = y_pred.replace('\n', '')
    y_true = y_true.replace('\r', '')
    y_pred = y_pred.replace('\r', '')

    # remove all semicolons
    y_true = y_true.replace(';', '')
    y_pred = y_pred.replace(';', '')

    # remove all commas
    y_true = y_true.replace(',', '')
    y_pred = y_pred.replace(',', '')

    # remove everything that is like T + number + . (up to T6)
    y_true = y_true.replace('t0.', '')
    y_pred = y_pred.replace('t0.', '')
    y_true = y_true.replace('t1.', '')
    y_pred = y_pred.replace('t1.', '')
    y_true = y_true.replace('t2.', '')
    y_pred = y_pred.replace('t2.', '')
    y_true = y_true.replace('t3.', '')
    y_pred = y_pred.replace('t3.', '')
    y_true = y_true.replace('t4.', '')
    y_pred = y_pred.replace('t4.', '')
    y_true = y_true.replace('t5.', '')
    y_pred = y_pred.replace('t5.', '')
    y_true = y_true.replace('t6.', '')
    y_pred = y_pred.replace('t6.', '')
    """
    
    if y_true == y_pred:
        return 1
    else:
        return 0

def reaarangetuple(tup, indices):
    '''
    Rearranges the tuple according to the indices

    :param tup: the tuple to be rearranged
    :param indices: the indices to rearrange the tuple

    :return: the rearranged tuple
    '''
    return tuple(tup[i] for i in indices)

def switch_place(y_pred):
    '''
    Checks if the result is a list of tuples
    If it is, the function returns a list of lists containing the tuples with the elements switched in all combinations possible
    If it is not, it only returns y_pred

    :param y_pred: the result to be checked
    '''
    length_of_tuple = len(y_pred[0])
    indices = range(length_of_tuple)
    if len(indices) >= 8:
        print("Too many indices to be permutating, returning only the original prediction, which means the prediction is counted as wrong.")
        return [y_pred]
    permutations_list = list(permutations(indices))
    new_y_pred = []
    for perm in permutations_list:
        new_y_pred_unit = []
        for tup in y_pred:
            new_y_pred_unit.append(reaarangetuple(tup, perm))
        new_y_pred.append(new_y_pred_unit)
    return new_y_pred


# closest we could get to the Spider evaluation metric for execution accuracy, link: https://ar5iv.labs.arxiv.org/html/1809.08887
def execution_accuracy(y_true, y_pred, dbpath):
    '''
    Function to return the execution accuracy of a query

    :param y_true: the true query
    :param y_pred: the predicted query
    :param dbpath: the path to the database

    :return: 1 if the true and predicted queries return the same result, 0 otherwise
    '''
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    # check if the predicted queries are correct
    try:
        cursor.execute(y_pred)
        pred_result = cursor.fetchall()
    except:
        if y_true == y_pred:
            return .3
        return 0
    
    try:
        cursor.execute(y_true)
        true_result = cursor.fetchall()
    except:
        print("True query failed, counting the prediction as correct as it correctly interrogated the database.")
        return 1

    if true_result == pred_result:
        if true_result == []:
            return .5
        return 1
    
    
    if type(pred_result) != list or pred_result == [] or type(pred_result[0]) != tuple or len(pred_result[0]) == 1:
        return 0
    
    else:
        list_pred = switch_place(pred_result)
        for pred_result in list_pred:
            if true_result == pred_result:
                if true_result == []:
                    return .5
                return 1
        return 0
### ---------------------------------------- ###
################################################
        

###########################################################
### Relic, used for the first iterations of the project ###
def sample_with_schema(sample):
    '''
    Function to return the sample with the schema concatenated to the sample
    with the database id removed

    :param sample: the sample to be concatenated with the schema

    :return: the sample with the schema concatenated
    '''
    from arguments import params
    args = params()
    databases_path = args.databases_path
    db_id = sample['db_id']
    with open(f'{databases_path}/database/{db_id}/schema.sql') as f:
        lines = f.readlines()
    new_sample = {}
    # concatenate the schema to a single string
    lines = ' '.join(lines)
    new_sample['schema'] = lines
    new_sample['query'] = sample['query']
    new_sample['question'] = sample['question']
    return new_sample
### --------------------------------------------------- ###
###########################################################


##############
### Logins ###
def hf_login_read():
    print("Logging in to HuggingFace")
    from arguments import params
    args = params()
    hf_token_read = args.hf_token_read
    from huggingface_hub import login

    login(
    token=hf_token_read,
    add_to_git_credential=True
    )
    print("Logged in to HuggingFace for read")
    
    return

def hf_login_write():
    print("Logging in to HuggingFace")
    from arguments import params
    args = params()
    hf_token_write = args.hf_token_write
    from huggingface_hub import login

    login(
    token=hf_token_write,
    add_to_git_credential=True
    )
    print("Logged in to HuggingFace for write")
### ------ ###
##############


##############################################
### Datasets loading and related utilities ###
def datasets_loading(prepare=False):
    print("Preparing the datasets")
    from arguments import params
    args = params()
    datasets_path = args.datasets_path
    if prepare:
        import dataset_prep as ds_p
        # Prepare the datasets
        train_dataset, dev_dataset, test_dataset, test_dataset_ids, train_dataset_ids = ds_p.dataset_preparation()


        # Save the datasets
        # Only do once
        train_dataset.to_json(f"{datasets_path}/train_dataset.json")
        dev_dataset.to_json(f"{datasets_path}/dev_dataset.json")
        test_dataset.to_json(f"{datasets_path}/test_dataset.json")
        import pandas as pd
        df = pd.DataFrame(test_dataset_ids)
        df.to_json(f"{datasets_path}/test_dataset_ids.json")
        df = pd.DataFrame(train_dataset_ids)
        df.to_json(f"{datasets_path}/train_dataset_ids.json")

    # Load the datasets
    from datasets import load_dataset
    train_dataset = load_dataset("json", data_files=f"{datasets_path}/train_dataset.json")
    dev_dataset = load_dataset("json", data_files=f"{datasets_path}/dev_dataset.json")
    test_dataset = load_dataset("json", data_files=f"{datasets_path}/test_dataset.json")
    test_dataset_ids = load_dataset("json", data_files=f"{datasets_path}/test_dataset_ids.json")
    train_dataset_ids = load_dataset("json", data_files=f"{datasets_path}/train_dataset_ids.json")
    print("Datasets are ready")
    
    return train_dataset['train'], dev_dataset['train'], test_dataset['train'], test_dataset_ids['train'], train_dataset_ids['train']


def ppo_dataset(dataset, model_name):
    '''
    Function to prepare the dataset for PPO training

    :param dataset: the dataset to be prepared

    :return: the prepared dataset
    '''
    # remove the columns that are not needed
    # during PPO, only the query is needed
    # as a reward model has already been trained

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{args.hf_username}/{model_name}")
    
    def prepare_sample(sample):
        if args.conv_roles == "user-assistant":
            query = tokenizer.apply_chat_template(sample["messages"][0:1], tokenize=False, add_generation_prompt=True)
            response = tokenizer.apply_chat_template(sample["messages"][1:2], tokenize=False, add_generation_prompt=False)
        elif args.conv_roles == "sys-user-assistant":
            query = tokenizer.apply_chat_template(sample["messages"][0:2], tokenize=False, add_generation_prompt=True)
            response = tokenizer.apply_chat_template(sample["messages"][2:3], tokenize=False, add_generation_prompt=False)
        return {"query": query, "true_response": response}

    ppo_dataset_raw = dataset.map(prepare_sample, remove_columns=dataset.features, batched=False)

    ppo_dataset = ppo_dataset_raw.map(lambda x: {"query": x["query"], "true_response": x["true_response"]})

    #print(f"PPO dataset: {ppo_dataset}")
    #print(f"PPO dataset sample: {ppo_dataset[0]}")
    #print(f"PPO dataset sample in natural language: {tokenizer.decode(ppo_dataset[0]['query']), tokenizer.decode(ppo_dataset[0]['true_response'])}")

    return ppo_dataset


def dpo_dataset():
    '''
    Function to load the DPO dataset from HuggingFace

    :return: the DPO dataset
    '''
    from datasets import load_dataset
    dpo_dataset = load_dataset(f"{args.hf_username}/spider-dpo-dataset")
    dpo_dataset_ids = load_dataset(f"{args.hf_username}/spider-dpo-dataset-ids")
    return dpo_dataset['train'], dpo_dataset_ids['train']['db_id']
### -------------------------------------- ###
##############################################

