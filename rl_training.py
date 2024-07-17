from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, setup_chat_format, DPOTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from arguments import params

from tqdm import tqdm
import time
from Levenshtein import distance as levenshtein_distance

args = params()


def train_ppo(model_name, ppo_dataset):
    '''
    Function to train the model with PPO

    :param ppo_dataset: the dataset to use for training

    :return: None
    '''

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )


    print("Loading model")

    # setup the LoRA config
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=32,
            bias="none",
            target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM", # for encoder only models like Llama or GPT
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        f"{args.hf_username}/{model_name}_full",
        device_map = "auto",
        torch_dtype=torch.bfloat16,
        quantization_config=nf4_config if args.quantize else None,
        attn_implementation="flash_attention_2",
        peft_config=peft_config,
    )

    ''' # Now unnecessary
    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Manually move all FP4 quantization layers to the correct device
    def move_quant_layers_to_device(model, device):
        for module in model.modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'quant_state'):
                module.cuda()

    move_quant_layers_to_device(model, device)
    '''
    tokenizer = AutoTokenizer.from_pretrained(f"{args.hf_username}/{model_name}_full")

    print("Model loaded. Model type: ", type(model))

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # tokenize the dataset
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        sample["tr_ids"] = tokenizer.encode(sample["true_response"])
        # pad the ids
        return sample
    
    # cut the dataset to the first 500 samples (randomly selected)
    ppo_dataset = ppo_dataset.shuffle(seed=args.seed)
    ppo_dataset = ppo_dataset.select(range(args.num_ppo if (args.num_ppo != -1 and args.num_ppo < len(ppo_dataset)) else len(ppo_dataset)))

    ppo_dataset = ppo_dataset.map(tokenize, batched=False)


    # datacollator for padding the dataset
    def custom_collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        tr_ids = [item['tr_ids'] for item in batch]
        return {
            'input_ids': tokenizer.pad({'input_ids': input_ids}, padding=True, return_tensors="pt")['input_ids'],
            'tr_ids': tokenizer.pad({'input_ids': tr_ids}, padding=True, return_tensors="pt")['input_ids'],
            'true_response': [item['true_response'] for item in batch],
            'query': [item['query'] for item in batch],
        }
    
    
    ### Setup chat format (from not finetuned model) ###
    #model, tokenizer = setup_chat_format(model, tokenizer)
    ### -------------------------------------------- ###


    # setup the PPO config
    config = PPOConfig(
        model_name=f"{args.hf_username}/{model_name}-ppo",
        remove_unused_columns=False,
        batch_size=16,
        mini_batch_size=1,
        gradient_accumulation_steps=16,
        kl_penalty="mse", # avoids negative KL values (due to approximations) for small batches,
        # which lead to the model trying to get those negative values to get good rewards,
        # instead of focusing on the task
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    

    # setup the PPO trainer
    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dataset=ppo_dataset,
        data_collator=custom_collate_fn,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )


    def reward_model(predictions, targets):
        '''
        Reward function for the PPO trainer

        Levenshtein distance is used as the reward function

        :param predictions: the predictions made by the model, a list of strings
        :param targets: the targets, a list of strings

        :return: the reward
        '''
        rewards = []
        for prediction, target in zip(predictions, targets):
            rewards.append(-levenshtein_distance(prediction, target))
        return rewards


    

    generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 256,
    "return_prompt": False,
    }



    epochs = 1
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(trainer.dataloader):
            # break the loop if the batch is None
            if batch is None:
                break

            start_time_big = time.time()
            query_tensors = batch["input_ids"]

            # removing the padding tokens at the beginning of the query tensors
            cleaned_query_tensors = []
            eos_token_id = tokenizer.eos_token_id
            for tensor in query_tensors:
                # Find the first index where the token is not eos_token_id
                not_eos_idx = (tensor != eos_token_id).nonzero(as_tuple=True)[0][0]
                # Append the sliced tensor from not_eos_idx to the end
                cleaned_query_tensors.append(tensor[not_eos_idx:])
        
            #### Get response from SFTModel
            print("Generating response")
            start_time = time.time()
            response_tensors = trainer.generate(cleaned_query_tensors, **generation_kwargs)
            print("Generation time: ", time.time() - start_time, " seconds")
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            batch["response"] = [r[:r.find("<|im_end|>")] for r in batch["response"]]
            print("Response: ", batch["response"])

        
            #### Compute reward score
            predictions = [r for r in batch["response"]]
            targets = [r for r in batch["true_response"]]
            pipe_outputs = reward_model(predictions, targets)
            #print("Pipe outputs: ", pipe_outputs)
            rewards = [torch.tensor(output, dtype=torch.float32) for output in pipe_outputs]
            

            #### Remove too long query-response couples
            # the max number of tokens in the query tensors must be 2300 to avoid CUDA OOM errors
            # but due to sharding with reserved but unallocated memory, we have to lower that limit
            # we remove the couples query-response that are too long
            too_long_idx_list = []
            for i in range(len(cleaned_query_tensors)):
                if len(cleaned_query_tensors[i]) > 2000:
                    too_long_idx_list.append(i)
            # reverse the list to avoid index errors
            too_long_idx_list.reverse()
            for idx in too_long_idx_list:
                cleaned_query_tensors[idx] = torch.tensor([tokenizer.eos_token_id])
                response_tensors[idx] = torch.tensor([tokenizer.eos_token_id])
                rewards[idx] = torch.tensor(0.0)
            
            print(f"Removed {len(too_long_idx_list)} couples query-response that are too long")
        

            #### Run PPO step
            if len(cleaned_query_tensors) > 0:
                cleaned_query_tensors = [q for q in cleaned_query_tensors]
                response_tensors = [r for r in response_tensors]
                # add the prompt tokens at the beginning of the response tensors
                response_tensors = [torch.cat([cleaned_query_tensors[i], response_tensors[i]]) for i in range(len(response_tensors))]
                stats = trainer.step(cleaned_query_tensors, response_tensors, rewards)
                trainer.log_stats(stats, batch, rewards)
        
            print("Batch time: ", time.time() - start_time_big, " seconds")


    # Save model
    trainer.save_pretrained(f"{model_name}-ppo")

    ### Free the memory again ###
    del model
    del trainer
    torch.cuda.empty_cache()
    ### --------------------- ###


    print("Merging PEFT and base model")
    ### Merge PEFT and base model ###

    model = AutoPeftModelForCausalLM.from_pretrained(
        f"{model_name}-ppo",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{model_name}-ppo", safe_serialization=True, max_shard_size="2GB")
    ### ------------------------ ###

    print("Pushing model to the hub")
    # push the model to the hub
    merged_model.push_to_hub(f"{args.hf_username}/{model_name}-ppo")
    print("Model pushed to the hub")


    return


def train_dpo(model_name, dpo_dataset):
    '''
    Function to train the model with DPO

    :param model_id: the model id to train
    :param model_name: the model name to save the model as
    :param dpo_dataset: the dataset to use for training

    :return: None
    '''

    # shuffle the dataset
    dpo_dataset = dpo_dataset.shuffle(seed=args.seed)

    # no DPOConfig in this version, it is in the dev version of trl even if HuggingFace says it has been working from some time
    
    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )


    print("Loading model")

    # setup the LoRA config
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=32,
            bias="none",
            target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM", # for encoder only models like Llama or GPT
    )

    model = AutoModelForCausalLM.from_pretrained(
        f"{args.hf_username}/{model_name}_full",
        device_map = "auto",
        torch_dtype=torch.bfloat16,
        quantization_config=nf4_config if args.quantize else None,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(f"{args.hf_username}/{model_name}_full")

    print("Model loaded. Model type: ", type(model))

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dpo_model_name = model_name + "-dpo"

    ### Training arguments ###
    from transformers import TrainingArguments
    
    train_args = TrainingArguments(
        output_dir=dpo_model_name,              # directory to save and repository id
        num_train_epochs=args.epochs_dpo,           # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-6,                     # learning rate
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",             # learning rate scheduler type
        push_to_hub=True,                       # push model to hub at every save
        evaluation_strategy="steps" if args.eval else 'no',
        eval_steps=100 if args.eval else None,
    )

    dpo_args = {
        "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
        "loss_type": "sigmoid"                  # The loss type for DPO.
    }
    ### ------------------- ###

    
    trainer = DPOTrainer(
        model,
        ref_model=None, # set to none since we use peft
        peft_config=peft_config,
        args=train_args,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        max_length=2816,
        max_prompt_length=2304,
        beta=dpo_args["beta"],
        loss_type=dpo_args["loss_type"],
    )


    trainer.train()

    trainer.save_model()

    del model
    del trainer
    torch.cuda.empty_cache()

    print("Merging PEFT and base model")
    ### Merge PEFT and base model ###

    model = AutoPeftModelForCausalLM.from_pretrained(
        dpo_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(dpo_model_name, safe_serialization=True, max_shard_size="2GB")
    ### ------------------------ ###

    print("Pushing model to the hub")
    # push the model to the hub
    merged_model.push_to_hub(f"{args.hf_username}/{dpo_model_name}_full")
    tokenizer.push_to_hub(f"{args.hf_username}/{dpo_model_name}_full")
    print("Model pushed to the hub")


    return

