import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
import utils
from arguments import params

from tqdm import tqdm
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets

args = params()
databases_path = args.databases_path

def dpo_dataset_create(model_name, train_dataset, train_dataset_ids):
    '''
    Function to prepare the DPO dataset
    '''
    print("Warning, dpo dataset already done!")
    print("If you want to remake the dpo dataset, modify the function in dpo_dataset_creation.py")
    print("Exiting")
    return

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load the model
    print("Loading model")
    peft_model_id = f"{args.hf_username}/{model_name}"
    model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=nf4_config if args.quantize else None,
    attn_implementation="flash_attention_2",
    )

    print("Model loaded. Model type: ", type(model))

    tokenizer = AutoTokenizer.from_pretrained(f"{args.hf_username}/{model_name}")
    # load into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    ### Quick test on sample ###
    from random import randint

    # random sample from test dataset
    rand_idx = randint(0, len(train_dataset))

    print(f"EOS token: {pipe.tokenizer.eos_token}")

    # Test on sample
    if args.conv_roles == "sys-user-assistant":
        prompt = pipe.tokenizer.apply_chat_template(train_dataset[rand_idx]["messages"][0:2], tokenize=False, add_generation_prompt=True)
    elif args.conv_roles == "user-assistant":
        prompt = pipe.tokenizer.apply_chat_template(train_dataset[rand_idx]["messages"][0:1], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    if args.conv_roles == "sys-user-assistant":
        print(f"Natural language request:\n{train_dataset[rand_idx]['messages'][1]['content']}\n\n")
        print(f"Original Answer:\n{train_dataset[rand_idx]['messages'][2]['content']}\n\n")
    elif args.conv_roles == "user-assistant":
        print(f"Natural language request:\n{train_dataset[rand_idx]['messages'][0]['content']}\n\n")
        print(f"Original Answer:\n{train_dataset[rand_idx]['messages'][1]['content']}\n\n")
    generated_answer = outputs[0]['generated_text'][len(prompt):].strip()
    print(f"First generated Answer:\n{generated_answer}\n\n")
    if generated_answer.find(";") != -1:
        generated_answer = generated_answer[:generated_answer.find(";")+1]
    print(f"Generated Answer:\n{generated_answer}\n\n")
    ### --------------------- ###


    ### Predict function ###
    def predict(sample):

        if args.conv_roles == "sys-user-assistant":
            prompt = pipe.tokenizer.apply_chat_template(sample["messages"][0:2], tokenize=False, add_generation_prompt=True)
            schema = sample["messages"][1]["content"]
        elif args.conv_roles == "user-assistant":
            prompt = pipe.tokenizer.apply_chat_template(sample["messages"][0:1], tokenize=False, add_generation_prompt=True)
            schema = sample["messages"][0]["content"]
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id, temperature=5.0, top_k=50, top_p=0.1)
        predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()

        true_predicted_answer = predicted_answer
        
        # remove the part after the first semicolon
        if predicted_answer.find(";") != -1:
            predicted_answer = predicted_answer[:predicted_answer.find(";")+1]


        # check first word, if it isn't SELECT, add it
        low_pred = predicted_answer.lower()
        if not low_pred.startswith("select") and not low_pred.startswith(" select"):
            predicted_answer = "SELECT " + predicted_answer


        return predicted_answer, true_predicted_answer
    ### ------------------ ###



    ### Prediction loop ###
    all_time_ex_acc = 0
    dpo_dataset_dict = {"chosen": [], "rejected": [], "prompt": []}
    dpo_db_ids = []
    for i in tqdm(range(len(train_dataset)), "Dataset sample: "):
        db_id = train_dataset_ids['0'][0][f"{i}"]
        db_path = f"{databases_path}/database/{db_id}/{db_id}.sqlite"
        prediction, raw_prediction = predict(train_dataset[i])
        print(f"Prediction: {prediction}")

        if args.conv_roles == "sys-user-assistant":
            true_pred = "SELECT " + train_dataset[i]["messages"][2]["content"]
            true_pred.replace("<|im_end|>", "")
            ex_acc = utils.execution_accuracy(true_pred, prediction, db_path)
            

        elif args.conv_roles == "user-assistant":
            true_pred = "SELECT " + train_dataset[i]["messages"][1]["content"]
            true_pred.replace("<|im_end|>", "")
            ex_acc = utils.execution_accuracy(true_pred, prediction, db_path)
        
        
        
        # check if None returned for both
        if ex_acc == .5:
            ex_acc = 1

        # check if database fails
        if ex_acc == .3:
            ex_acc = 1
        
    
        all_time_ex_acc += ex_acc
        all_time_ex_acc /= i+1
        print(f"Execution accuracy: {all_time_ex_acc}")
        all_time_ex_acc *= i+1

        if ex_acc != 1:
            if args.conv_roles == "sys-user-assistant":
                system_message = train_dataset[i]["messages"][0]
                user_message = train_dataset[i]["messages"][1]
                true_pred = [train_dataset[i]["messages"][2]]
                prediction = [{"role": "assistant", "content": raw_prediction}]
                prompt = tokenizer.apply_chat_template([system_message, user_message], tokenize=False)
                message_chosen = tokenizer.apply_chat_template(true_pred, tokenize=False)
                message_rejected = tokenizer.apply_chat_template(prediction, tokenize=False)
            elif args.conv_roles == "user-assistant":
                user_message = [train_dataset[i]["messages"][0]]
                true_pred = [train_dataset[i]["messages"][1]]
                prediction = [{"role": "assistant", "content": raw_prediction}]
                prompt = tokenizer.apply_chat_template(user_message, tokenize=False)
                message_chosen = tokenizer.apply_chat_template(true_pred, tokenize=False)
                message_rejected = tokenizer.apply_chat_template(prediction, tokenize=False)

            dpo_db_ids.append(db_id)
            dpo_dataset_dict["prompt"].append(prompt)
            dpo_dataset_dict["chosen"].append(message_chosen)
            dpo_dataset_dict["rejected"].append(message_rejected)
        
        # save every 500 samples, as scaleway likes to disconnect
        if i % 500 == 0 and i != 0:
            dpo_dataset = DatasetDict({"train": Dataset.from_dict(dpo_dataset_dict)})
            dpo_dataset_ids = DatasetDict( {"train": Dataset.from_dict( {"db_id": dpo_db_ids} ) } )
            dpo_dataset.save_to_disk("spider_dpo_dataset_temp")
            dpo_dataset_ids.save_to_disk("spider_dpo_dataset_ids_temp")
            dpo_dataset.push_to_hub(f"{args.hf_username}/spider-dpo-dataset-temp")
            dpo_dataset_ids.push_to_hub(f"{args.hf_username}/spider-dpo-dataset-ids-temp")

            # reset dict
            dpo_dataset_dict = {"chosen": [], "rejected": [], "prompt": []}
            dpo_db_ids = []
        
            if i == 500:
                dpo_dataset.push_to_hub(f"{args.hf_username}/spider-dpo-dataset")
                dpo_dataset_ids.push_to_hub(f"{args.hf_username}/spider-dpo-dataset-ids")
            
            else:
                # load temp dataset and full dataset, combine them, and save
                temp_dpo_dataset = load_dataset(f"{args.hf_username}/spider-dpo-dataset-temp", split="train")
                temp_dpo_dataset_ids = load_dataset(f"{args.hf_username}/spider-dpo-dataset-ids-temp", split="train")
                full_dpo_dataset = load_dataset(f"{args.hf_username}/spider-dpo-dataset", split="train")
                full_dpo_dataset_ids = load_dataset(f"{args.hf_username}/spider-dpo-dataset-ids", split="train")

                dpo_dataset = concatenate_datasets([temp_dpo_dataset, full_dpo_dataset])
                dpo_dataset_ids = concatenate_datasets([temp_dpo_dataset_ids, full_dpo_dataset_ids])

                dpo_dataset = DatasetDict({"train": dpo_dataset})
                dpo_dataset_ids = DatasetDict({"train": dpo_dataset_ids})

                dpo_dataset.push_to_hub(f"{args.hf_username}/spider-dpo-dataset")
                dpo_dataset_ids.push_to_hub(f"{args.hf_username}/spider-dpo-dataset-ids")




    # final concatenation and saving
    dpo_dataset = Dataset.from_dict(dpo_dataset_dict)
    dpo_dataset_ids = Dataset.from_dict( {"db_id": dpo_db_ids} )
    full_dpo_dataset = load_dataset(f"{args.hf_username}/spider-dpo-dataset", split="train")
    full_dpo_dataset_ids = load_dataset(f"{args.hf_username}/spider-dpo-dataset-ids", split="train")
    dpo_dataset = concatenate_datasets([dpo_dataset, full_dpo_dataset])
    dpo_dataset_ids = concatenate_datasets([dpo_dataset_ids, full_dpo_dataset_ids])

    dpo_dataset = DatasetDict({"train": dpo_dataset})
    dpo_dataset_ids = DatasetDict({"train": dpo_dataset_ids})

    dpo_dataset.save_to_disk("spider_dpo_dataset")
    dpo_dataset_ids.save_to_disk("spider_dpo_dataset_ids")
    dpo_dataset.push_to_hub(f"{args.hf_username}/spider-dpo-dataset")
    dpo_dataset_ids.push_to_hub(f"{args.hf_username}/spider-dpo-dataset-ids")


    return

    