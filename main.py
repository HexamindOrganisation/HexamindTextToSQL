import utils
import evaluate_model
import training
import arguments
import rl_training
import dpo_dataset_creation



if __name__ == "__main__":

    # get arguments
    print("Getting arguments")
    args = arguments.params()
    model_id = args.model_to_train # hf path
    model_name = args.model_name # just the name, not the path
    print("--------done--------")

    # connection to HuggingFace for reading purposes
    print("Logging in to HuggingFace for reading purposes")
    utils.hf_login_read()
    print("--------done--------")

    # connection to HuggingFace for writing purposes
    print("Logging in to HuggingFace for writing purposes")
    utils.hf_login_write()
    print("--------done--------")

    # load datasets
    print("Loading datasets")
    train_dataset, dev_dataset, test_dataset, test_dataset_ids, train_dataset_ids = utils.datasets_loading(prepare=args.prepare_datasets)
    print("--------done--------")

    if args.create_dpo_dataset:
        # prepare the DPO dataset
        print("Preparing the DPO dataset")
        dpo_dataset_creation.dpo_dataset_create(model_name, train_dataset, train_dataset_ids)
        print("--------done--------")

    if args.train:
        # train the model
        print("Training the model")
        training.train(model_id, model_name, train_dataset, dev_dataset)
        print("--------done--------")
    
    
    if args.train_dpo:
        # train the model with DPO
        print("Training the model with DPO")
        print("Loading the DPO dataset")
        dpo_dataset, dpo_dataset_ids = utils.dpo_dataset()
        print("DPO dataset loaded")
        rl_training.train_dpo(model_name, dpo_dataset)
        print("--------done--------")
    
    
    if args.train_ppo:
        # train the model with PPO
        print("Training the model with PPO")
        ppo_dataset = utils.ppo_dataset(train_dataset, model_name)
        rl_training.train_ppo(model_name, ppo_dataset)
        print("--------done--------")

    if args.test:
        # evaluate the model
        print("Evaluating the model")
        execution_accuracy, exact_match_accuracy = evaluate_model.evaluation(model_name, test_dataset, test_dataset_ids)
        print("--------done--------")

        # log the results
        print("Logging the results")
        print(f"Execution Accuracy: {execution_accuracy}")
        print(f"Exact Match Accuracy: {exact_match_accuracy}")
        print("--------done--------")


    print("All done!")



