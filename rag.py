import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from arguments import params

args = params()

def reformat_sample(sample):
    """
    Reformat a sample to a string
    """

    if args.conv_roles == "sys-user-assistant":
        schema_and_question = sample['messages'][1]['content']
    elif args.conv_roles == "user-assistant":
        schema_and_question = sample['messages'][0]['content']
    else:
        raise ValueError("Invalid conversation roles")

    if args.rag_input_format == "schema_and_question":
        return schema_and_question
    elif args.rag_input_format == "question":
        split_index = schema_and_question.find("--")
        question = schema_and_question[split_index+2:]
        return question
    
    raise ValueError("Invalid RAG input format")

def create_embeddings(dataset):
    """
    Create embeddings for the dataset
    """

    # reformat the dataset
    dataset_reformatted = [reformat_sample(sample) for sample in dataset]

    # create the embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(dataset_reformatted)

    return embeddings, vectorizer

def get_most_similar_samples(sample, embeddings, vectorizer, dataset, nb_samples=args.nb_rag_samples):
    """
    Get the most similar samples to the given sample

    Deterministic rag strategy
    """

    # reformat the sample
    sample_reformatted = reformat_sample(sample)

    # create the embeddings for the sample
    sample_embedding = vectorizer.transform([sample_reformatted])

    # calculate the cosine similarity
    similarity = cosine_similarity(sample_embedding, embeddings)

    # get the indices of the most similar samples
    indices = np.argsort(similarity[0])[::-1][:nb_samples]

    # get the most similar samples
    most_similar_samples = [dataset[int(i)] for i in indices]

    return most_similar_samples




