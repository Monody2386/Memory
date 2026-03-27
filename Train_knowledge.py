from knowledge.train_knowledge import *  # noqa: F401,F403


if __name__ == "__main__":
    from knowledge.train_knowledge import predict_next_word, train_via_feed_relations
    from knowledge.training import run_long_training_and_save

    relations = [("apple", "fruit", "include")]
    train_via_feed_relations(relations)
    print(predict_next_word("banana", "include"))
    run_long_training_and_save()
