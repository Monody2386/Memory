from knowledge.train_adj_knowledge import *  # noqa: F401,F403


if __name__ == "__main__":
    from knowledge.train_adj_knowledge import DEFAULT_ADJ_TRAINING_DATA_PATH, train_adj_from_file

    results, average_loss = train_adj_from_file()
    print(f"Trained {len(results)} adj-noun triples from {DEFAULT_ADJ_TRAINING_DATA_PATH}")
    for result in results:
        print(result)
    print({"average_loss": average_loss})
