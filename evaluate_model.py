import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline, AutoModelForCausalLM
import utils
from arguments import params
import time
import datetime
import json

args = params()
databases_path = args.databases_path

def evaluation(model_name, test_dataset, test_dataset_ids):

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading model")
    peft_model_id = f"{args.hf_username}/{model_name}"

    if args.test_ppo:
        peft_model_id += "-ppo"
    
    elif args.test_dpo:
        peft_model_id += "-dpo"


    ### Model loading ###
    if args.test_ppo:
        # Load PPO model
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=nf4_config if args.quantize else None,
        attn_implementation="flash_attention_2",
        )
    
    elif args.test_dpo:
        # Load DPO model
        model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=nf4_config if args.quantize else None,
        attn_implementation="flash_attention_2",
        )
    
    else:
        # Load PEFT adapter
        model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=nf4_config if args.quantize else None,
        attn_implementation="flash_attention_2",
        )
    ### ---------------- ###


    print("Model loaded. Model type: ", type(model))

    tokenizer = AutoTokenizer.from_pretrained(f"{args.hf_username}/{model_name}")
    # load into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    
    ### Quick test on sample ###
    from random import randint

    # random sample from test dataset
    rand_idx = randint(0, len(test_dataset))

    print(f"EOS token: {pipe.tokenizer.eos_token}")

    # Test on sample
    if args.conv_roles == "sys-user-assistant":
        prompt = pipe.tokenizer.apply_chat_template(test_dataset[rand_idx]["messages"][0:2], tokenize=False, add_generation_prompt=True)
    elif args.conv_roles == "user-assistant":
        prompt = pipe.tokenizer.apply_chat_template(test_dataset[rand_idx]["messages"][0:1], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    if args.conv_roles == "sys-user-assistant":
        print(f"Natural language request:\n{test_dataset[rand_idx]['messages'][1]['content']}\n\n")
        print(f"Original Answer:\n{test_dataset[rand_idx]['messages'][2]['content']}\n\n")
    elif args.conv_roles == "user-assistant":
        print(f"Natural language request:\n{test_dataset[rand_idx]['messages'][0]['content']}\n\n")
        print(f"Original Answer:\n{test_dataset[rand_idx]['messages'][1]['content']}\n\n")
    generated_answer = outputs[0]['generated_text'][len(prompt):].strip()
    print(f"First generated Answer:\n{generated_answer}\n\n")
    if generated_answer.find(";") != -1:
        generated_answer = generated_answer[:generated_answer.find(";")+1]
    print(f"Generated Answer:\n{generated_answer}\n\n")
    ### --------------------- ###

    # load special model for CoT
    if args.use_schema_linking:
        print("Loading CoT model")
        cot_model = AutoModelForCausalLM.from_pretrained(
            args.cot_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=nf4_config if args.quantize else None,
            attn_implementation="flash_attention_2",
        )
        cot_tokenizer = AutoTokenizer.from_pretrained(args.cot_model_id)
        cot_pipe = pipeline("text-generation", model=cot_model, tokenizer=cot_tokenizer)
        print("CoT model loaded")



    ### Evaluate the model ###

    ### Predict function ###
    def predict(sample):


        ## Chain of thought Schema Linking ##
        if args.use_schema_linking:
            
            schema_linking_prompt = "Give me the CREATE statements amongst the previous ones that are necessary to generate the SQL to answer the question.\n"

            schema_linking_system = """You will be given a SQL schema and a question in natural language.
Return the whole tables that are necessary to answer the question in SQL format, and don't answer the question.
"""

            user_beginning = "Here is the schema of the tables you will be working with:"

            if args.conv_roles == "sys-user-assistant":
                sample["messages"][0]["content"] = schema_linking_system
                schema_and_question = sample["messages"][1]["content"]
                question = schema_and_question[schema_and_question.find("--"):]
                # remove the SELECT
                schema_and_question = schema_and_question[:schema_and_question.find("--")]
                schema_and_question = schema_and_question + schema_linking_prompt
                sample["messages"][1]["content"] = schema_and_question
                prompt = cot_pipe.tokenizer.apply_chat_template(sample["messages"][0:2], tokenize=False, add_generation_prompt=True)
            elif args.conv_roles == "user-assistant":
                schema_and_question = sample["messages"][0]["content"]
                # remove Here is the schema of the tables you will be working with:
                table_schema_begin = schema_and_question.find(user_beginning)
                schema_and_question, text2sql_system = schema_and_question[table_schema_begin:], schema_and_question[:table_schema_begin]
                question = schema_and_question[schema_and_question.find("--"):]
                # remove the SELECT
                schema_and_question = schema_and_question[:schema_and_question.find("--")]
                schema_and_question = schema_and_question + schema_linking_prompt
                sample["messages"][0]["content"] = schema_linking_system + schema_and_question
                prompt = cot_pipe.tokenizer.apply_chat_template(sample["messages"][0:1], tokenize=False, add_generation_prompt=True)
            
            outputs = cot_pipe(prompt, max_new_tokens=256, do_sample=False, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
            new_schema = outputs[0]['generated_text'][len(prompt):].strip()

            # remove everything after assistant
            new_schema = new_schema[:new_schema.find("assistant")]
            # remove everything before CREATE
            new_schema = new_schema[new_schema.find("CREATE TABLE"):]
            # find the last CREATE
            last_create = new_schema.rfind("CREATE TABLE")
            # find the first semicolon after the last CREATE
            first_semicolon = new_schema[last_create:].find(";") + last_create
            # remove everything after the first semicolon after the last CREATE
            new_schema = new_schema[:first_semicolon+1]

            # get the tables from the schema
            tables = utils.get_table_names(new_schema)
            # get the lines in schema_and_question that begin with INSERT INTO
            inserts = []
            for line in schema_and_question.split("\n"):
                if line.lower().find("insert into ") != -1:
                    table = line[line.lower().find("insert into ")+len("insert into "):]
                    table = table[:table.lower().find("values")]
                    table = table.replace('"', "")
                    table = table.replace(" ", "")
                    if table.lower() in tables:
                        inserts.append(line)
            # add the inserts to the new schema
            for insert in inserts:
                new_schema += "\n" + insert

            print(f"New schema: {new_schema}")

            if args.conv_roles == "sys-user-assistant":
                sample["messages"][1]["content"] = new_schema + question
            elif args.conv_roles == "user-assistant":
                sample["messages"][0]["content"] = text2sql_system + new_schema + question
        ## ---------------- ##


        if args.conv_roles == "sys-user-assistant":
            prompt = pipe.tokenizer.apply_chat_template(sample["messages"][0:2], tokenize=False, add_generation_prompt=True)
            schema = sample["messages"][1]["content"]
        elif args.conv_roles == "user-assistant":
            prompt = pipe.tokenizer.apply_chat_template(sample["messages"][0:1], tokenize=False, add_generation_prompt=True)
            schema = sample["messages"][0]["content"]
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
        predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
        
        # remove the part after the first semicolon
        if predicted_answer.find(";") != -1:
            predicted_answer = predicted_answer[:predicted_answer.find(";")+1]


        ## Chain-of-thought Rectification ##
        if args.use_cot_rectification:

            print(f"\nFirst prediction: {predicted_answer}")
            
            # cot prompt
            cot_prompt = " " + predicted_answer
            cot_prompt += "\nCorrect the SQL query above if necessary. You can look for issues in the table names, column names, or the query itself.\nSELECT "           

            if args.conv_roles == "sys-user-assistant":
                sample["messages"][1]["content"] = sample["messages"][1]["content"] + cot_prompt
                prompt = pipe.tokenizer.apply_chat_template(sample["messages"][0:2], tokenize=False, add_generation_prompt=True)
                schema = sample["messages"][1]["content"]
            elif args.conv_roles == "user-assistant":
                sample["messages"][0]["content"] = sample["messages"][0]["content"] + cot_prompt
                prompt = pipe.tokenizer.apply_chat_template(sample["messages"][0:1], tokenize=False, add_generation_prompt=True)
                schema = sample["messages"][0]["content"]
            
            outputs = pipe(prompt, max_new_tokens=256, do_sample=False, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
            predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()

            print(f"Chain of thought prediction: {predicted_answer}\n")
        ## ---------------- ##
            

        # check first word, if it isn't SELECT, add it
        low_pred = predicted_answer.lower()
        if not low_pred.startswith("select") and not low_pred.startswith(" select"):
            predicted_answer = "SELECT " + predicted_answer

        
        # replace wrong table names with correct ones
        if not utils.check_table_names(predicted_answer, schema):
            #print(f"\nWrong table names in the predicted answer: {predicted_answer}")
            predicted_answer = utils.correct_query(predicted_answer, schema)
            #print(f"Corrected table names in the predicted answer: {predicted_answer}")


        return predicted_answer
    ### ------------------ ###

    success_rate_ex = []
    success_rate_em = []

    fail_ex_ids = []
    failed_questions = []
    fail_answers = []
    true_answers = []

    none_rate = 0
    database_fails = 0
    
    # sample test_dataset
    if args.num_tests == -1:
        test_dataset = test_dataset
    else:
        if args.num_tests > len(test_dataset):
            print(f"Number of tests ({args.num_tests}) is greater than the number of samples in the test dataset ({len(test_dataset)}).")
            print(f"Setting number of tests to the number of samples in the test dataset.")
            args.num_tests = len(test_dataset)
        indices = list(range(len(test_dataset)))
        import random
        # set seed
        random.seed(args.seed)
        random.shuffle(indices)
        # shuffle the dataset
        test_dataset = test_dataset.select(indices)
        test_dataset_ids_shuffled = {'0': [{}]}
        for j in range(len(test_dataset)):
            test_dataset_ids_shuffled['0'][0][f"{j}"] = test_dataset_ids['0'][0][f"{indices[j]}"]
        test_dataset_ids = test_dataset_ids_shuffled
        test_dataset = test_dataset.select(range(args.num_tests))
    print(f"Testing on {len(test_dataset)} samples")

    import sys
    # iterate over eval dataset and predict


    start_time = time.time()
    mean_time = 0
    for i in range(len(test_dataset)):
        db_id = test_dataset_ids['0'][0][f"{i}"]
        db_path = f"{databases_path}/test_database/{db_id}/{db_id}.sqlite"
        prediction = predict(test_dataset[i])
        if args.conv_roles == "sys-user-assistant":
            true_pred = "SELECT " + test_dataset[i]["messages"][2]["content"]
            true_pred.replace("<|im_end|>", "")

            success_rate_ex.append(utils.execution_accuracy(true_pred, prediction, db_path))
            success_rate_em.append(utils.exact_match_accuracy(true_pred, prediction))
        elif args.conv_roles == "user-assistant":
            true_pred = "SELECT " + test_dataset[i]["messages"][1]["content"]
            true_pred.replace("<|im_end|>", "")

            success_rate_ex.append(utils.execution_accuracy(true_pred, prediction, db_path))
            success_rate_em.append(utils.exact_match_accuracy(true_pred, prediction))
        
        # check if None returned for both
        if success_rate_ex[-1] == .5:
            success_rate_ex[-1] = 1
            none_rate += 1

        # check if database fails
        if success_rate_ex[-1] == .3:
            success_rate_ex[-1] = 1
            database_fails += 1
        
        # get the failed examples
        if success_rate_ex[-1] == 0:
            fail_ex_ids.append(db_id)
            fail_answers.append(prediction)
            if args.conv_roles == "sys-user-assistant":
                failed_questions.append(test_dataset[i]["messages"][1]["content"])
                true_answers.append("SELECT " + test_dataset[i]["messages"][2]["content"])
            elif args.conv_roles == "user-assistant":
                failed_questions.append(test_dataset[i]["messages"][0]["content"])
                true_answers.append("SELECT " + test_dataset[i]["messages"][1]["content"])
        



        execution_accuracy = sum(success_rate_ex)/len(success_rate_ex) * 1000 // 1 / 1000
        exact_match_accuracy = sum(success_rate_em)/len(success_rate_em) * 1000 // 1 / 1000
        completion_rate = (i+1)/len(test_dataset) * 1000 // 1 / 1000
        mean_time = (time.time() - start_time) / (i+1)
        estimated_time = mean_time * (len(test_dataset) - i - 1) /3600 * 1000 //1 /1000 # estimated time left in hours
        none_rate_true = none_rate / len(success_rate_ex) * 1000 // 1 / 1000
        msg = "Completion rate: {0}, Execution Accuracy: {1}, Exact Match Accuracy: {2}, Estimated time left: {3} hours, None rate for both: {4}, Database fails: {5}, ...".format(completion_rate,
                                                                                                    execution_accuracy,
                                                                                                    exact_match_accuracy,
                                                                                                    estimated_time,
                                                                                                    none_rate_true,
                                                                                                    database_fails)
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()

    # fails to the same csv file
    if len(fail_ex_ids) > 0:
        import pandas as pd

        failed_questions_cleaned = []
        # only take what is after '--'
        for q in failed_questions:
            failed_question_clean = ""
            __isin = True
            while __isin:
                if q.find("--") != -1:
                    q = q[q.find("--")+2:]
                    next_line = q.find("\n")
                    if next_line != -1:
                        failed_question_clean += q[:next_line] + ",\n"
                        q = q[next_line+1:]
                    else:
                        failed_question_clean += q
                        __isin = False
                else:
                    __isin = False
            failed_questions_cleaned.append(failed_question_clean)


        df = pd.DataFrame({"db_id": fail_ex_ids, "question": failed_questions_cleaned, "predicted_answer": fail_answers, "true_answer": true_answers})
        csv_name = f"failed_examples_{model_name}"
        if args.test_ppo:
            csv_name += "_ppo" 
        if args.test_dpo:
            csv_name += "_dpo"
        if args.use_schema_linking:
            csv_name += "_schema_linking"
        if args.use_cot_rectification:
            csv_name += "_cot_rectification"
        if args.use_rag:
            csv_name += "_rag"
            csv_name += f"{args.nb_rag_samples}"
        # add date and time
        date_and_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        csv_name += f"_{date_and_time}"

        df.to_csv(f"{csv_name}.csv", index=False)

    end_time = time.time()
    print(f"\nTime taken for evaluation: {end_time - start_time} seconds, or {(end_time - start_time)/60} minutes, or {(end_time - start_time)/3600} hours")



    # compute accuracy
    execution_accuracy = sum(success_rate_ex)/len(success_rate_ex)
    exact_match_accuracy = sum(success_rate_em)/len(success_rate_em)
    ### ------------------ ###

    print(f"Execution Accuracy: {execution_accuracy}")
    print(f"Exact Match Accuracy: {exact_match_accuracy}")

    # load everything into a json file
    file_name = model_name + "_ppo" if args.test_ppo else model_name
    if args.test_dpo:
        file_name += "_dpo"
    if args.use_schema_linking:
        file_name += "_schema_linking"
    if args.use_cot_rectification:
        file_name += "_cot_rectification"
    if args.use_rag:
        file_name += "_rag"
        file_name += f"{args.nb_rag_samples}"

    # add date and time
    date_and_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    file_name += f"_{date_and_time}"
        
    with open(f"evaluation_{file_name}_{args.num_tests}_tests.json", "w") as f:
        json.dump({"execution_accuracy": execution_accuracy, "exact_match_accuracy": exact_match_accuracy, "time taken in seconds": end_time - start_time,\
                    "time taken in minutes": (end_time - start_time)/60, "time taken in hours": (end_time - start_time)/3600}, f)

    return execution_accuracy, exact_match_accuracy




