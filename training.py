import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, AutoPeftModelForCausalLM
from arguments import params

args = params()

def train(model_id, model_name, train_dataset, dev_dataset):
    '''
    Function to train the model
    Saves the model to the hub

    :param model_id: the model id to be used for training
    :param model_name: the name of the model to be saved
    :param train_dataset: the training dataset

    :return: None
    '''


    ### BitAndBytes configuration ###
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    ### ------------------------- ###


    ### Peft model id ###
    """
    model_id = f"{args.hf_username}/{model_name}"

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    model = model.merge_and_unload()
    """
    ### -------------- ###


    ### Load model and tokenizer ###

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right' # to prevent warnings
    ### ------------------------ ###


    ### Setup chat format (from not finetuned model) ###
    model, tokenizer = setup_chat_format(model, tokenizer)
    ### -------------------------------------------- ###


    ### LoRA config ###
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM", # for encoder only models like Llama or GPT
    )
    ### ----------- ###



    ### Training arguments ###
    from transformers import TrainingArguments

    train_args = TrainingArguments(
        output_dir=model_name,                  # directory to save and repository id
        num_train_epochs=args.epochs,           # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",             # learning rate scheduler type
        push_to_hub=True,                       # push model to hub at every save
        evaluation_strategy="steps" if args.eval else 'no',
        eval_steps=100 if args.eval else None,
    )
    ### ------------------- ###


    ### Train the model ###
    from trl import SFTTrainer

    # max sequence length for model and packing of the dataset
    max_seq_length = 3072
    
    # shuffling and selecting subset
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if args.num_train == -1:
        train_dataset = train_dataset
    else:
        if args.num_train > len(train_dataset):
            print(f"Number of train samples to use ({args.num_train}) is greater than the number of samples in the train dataset ({len(train_dataset)}).")
            print(f"Setting number of train samples to the number of samples in the train dataset.")
            args.num_train = len(train_dataset)
        train_dataset = train_dataset.select(range(args.num_train))
        
    print("dataset length: ", len(train_dataset))

    # check max number of tokens in each sample
    import numpy as np
    print("Mean number of tokens in each sample: ", np.mean([len(tokenizer.tokenize(sample["messages"][0]['content'])) for sample in train_dataset]))

    
    #response_template = "SELECT"
    #collator = DataCollatorForCompletionOnlyLM(
    #    tokenizer=tokenizer,
    #    response_template=response_template,
    #)

    eos_token = tokenizer.eos_token
    print("EOS token id: ", eos_token)

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset if args.eval else None,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        # data_collator=collator, # using it leads to the model not stopping after generating one SQL statement
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save model
    trainer.save_model()
    ### ----------------- ###


    ### Free the memory again ###
    del model
    del trainer
    torch.cuda.empty_cache()
    ### --------------------- ###


    print("Merging PEFT and base model")
    ### Merge PEFT and base model ###

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(model_name, safe_serialization=True, max_shard_size="2GB")
    ### ------------------------ ###

    print("Pushing model to the hub")
    # push the model to the hub
    merged_model.push_to_hub(f"{args.hf_username}/{model_name}_full")
    tokenizer.push_to_hub(f"{args.hf_username}/{model_name}_full")
    print("Model pushed to the hub")


    return


