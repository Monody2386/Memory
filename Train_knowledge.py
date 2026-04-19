from knowledge.train_knowledge import *  # noqa: F401,F403


if __name__ == "__main__":
    from knowledge.train_knowledge import DEFAULT_TRAINING_DATA_PATH, train_from_file

    results = train_from_file()
    print(f"Trained {len(results)} triples from {DEFAULT_TRAINING_DATA_PATH}")
    for result in results:
        print(result)
