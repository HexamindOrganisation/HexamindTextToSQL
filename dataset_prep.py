from datasets import load_dataset, concatenate_datasets
from arguments import params
args = params()
import rag

def dataset_preparation():
    system_message = """You are a helpful assistant who answers questions about database tables by responding with SQL queries.
Users will provide you with a set of tables represented as CREATE TABLE statements followed by INSERT statements containing information about the data types contained in each of the tables.
Then users will ask a question.
Answer the user question by writing a SQL statement with the help of the provided SQL tables."""

    user_message = """Here is the schema of the tables you will be working with:
{schema}

-- {question} 
SELECT"""

    #eos_token = "<|im_end|>"

    def create_conversation(sample):
        query = sample["query"]
        # remove the first SELECT statement
        lower_query = query.lower()
        select_idx = lower_query.find("select") + 6
        query = query[select_idx:]
        if args.conv_roles == "sys-user-assistant":
            return {
                "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message.format(schema=sample["schema"], question=sample["question"])},
                {"role": "assistant", "content": query}
                ]
            }
        elif args.conv_roles == "user-assistant":
            return {
                "messages": [
                {"role": "user", "content": system_message + "\n" + user_message.format(schema=sample["schema"], question=sample["question"])},
                {"role": "assistant", "content": query}
                ]
            }
        
    def create_conversation_rag(sample):
        embeddings, vectorizer = rag.create_embeddings(train_dataset)
        most_similar_samples = rag.get_most_similar_samples(sample, embeddings, vectorizer, train_dataset)

        schema_and_question = sample['messages'][1]['content'] if args.conv_roles == "sys-user-assistant" else sample['messages'][0]['content']

        sys_prompt = ""

        if args.conv_roles == "user-assistant":
            table_schema_begin = schema_and_question.find("Here is the schema of the tables you will be working with:")
            schema_and_question, sys_prompt = schema_and_question[table_schema_begin:], schema_and_question[:table_schema_begin]

        examples_prompt = "Here are a few examples of tables, user input in natural language, and the expected query in SQL:\n"
        
        final_prompt = sys_prompt + examples_prompt
        for example in most_similar_samples:
            final_prompt += example["messages"][1]["content"]
            if args.conv_roles == "sys-user-assistant":
                final_prompt += example["messages"][2]["content"] + "\n\n\n"
            else:
                raise ValueError("Invalid conversation roles, you must modify the few prompt method to account for this.")
        
        final_prompt += schema_and_question + "\n\n"

        query = sample["messages"][2]["content"] if args.conv_roles == "sys-user-assistant" else sample["messages"][1]["content"]
        user_message = final_prompt

        # remove all occurences of "Here is the schema of the tables you will be working with:"
        user_message = user_message.replace("Here is the schema of the tables you will be working with:", "\n\n")
        
        if args.conv_roles == "sys-user-assistant":
            return {
                "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": query}
                ]
            }
        elif args.conv_roles == "user-assistant":
            return {
                "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": query}
                ]
            }

        



    # Load dataset from the hub
    dataset = load_dataset(args.dataset_on_hub)
    if args.use_gretel:
        gretel_dataset = load_dataset("gretelai/synthetic_text_to_sql")
        train_gretel_dataset = gretel_dataset["train"]
        train_gretel_dataset = train_gretel_dataset.map(lambda x: { "db_id": x["id"], "query": x["sql"], "question": x["sql_prompt"], "schema": x["sql_context"]})
        pd = train_gretel_dataset.to_pandas()
        pd = pd[["db_id", "query", "question", "schema"]]
        train_gretel_dataset = train_gretel_dataset.from_pandas(pd)
        train_gretel_dataset.shuffle(seed=args.seed)
        if args.nb_gretel_samples != -1 and args.nb_gretel_samples < len(train_gretel_dataset):
            train_gretel_dataset = train_gretel_dataset.select(range(args.nb_gretel_samples))

    print("Dataset loaded from HuggingFace")


    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    test_dataset = dataset["test"]

    test_dataset_ids = test_dataset["db_id"]
    train_dataset_ids = train_dataset["db_id"]

   
    train_dataset = train_dataset.map(create_conversation, remove_columns=train_dataset.features, batched=False)
    dev_dataset = dev_dataset.map(create_conversation, remove_columns=dev_dataset.features, batched=False)
    test_dataset = test_dataset.map(create_conversation, remove_columns=test_dataset.features, batched=False)

    if args.use_gretel:
        train_gretel_dataset = train_gretel_dataset.map(create_conversation, remove_columns=train_gretel_dataset.features, batched=False)
        train_dataset = concatenate_datasets([train_dataset, train_gretel_dataset])

    

    if args.use_rag:
        test_dataset = test_dataset.map(create_conversation_rag, remove_columns=test_dataset.features, batched=False)

        # show one sample
        print("Sample when using RAG:")
        print(test_dataset[0]["messages"][1]["content"])
        print("--------done--------")

    return train_dataset, dev_dataset, test_dataset, test_dataset_ids, train_dataset_ids