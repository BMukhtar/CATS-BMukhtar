import tensorflow as tf

def deserialize_data(example_proto):
    context_features = {
        "seg_lab": tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    sequence_features = {
        "true_seqs": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
        "fake_seqs": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    features = dict(sequence_parsed, **context_parsed)  # merging the two dicts
    return features
def deserialize_tf_record(record):
    """Deserializes a TFRecord into a dictionary.

    Args:
      record: A TFRecord.

    Returns:
      A dictionary of the TFRecord's features.
    """
    context_features = {
        "seg_lab": tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    sequence_features = {
        "true_seqs": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
        "fake_seqs": tf.io.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=record,
        context_features=context_features,
        sequence_features=sequence_features
    )

    features = dict(sequence_parsed, **context_parsed)  # merging the two dicts
    return features

def get_dataset(filename):
    """Creates a TF Dataset from a TFRecord file.

    Args:
      filename: The path to the TFRecord file.

    Returns:
      A TF Dataset.
    """
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(deserialize_tf_record)
    return dataset

dataset = get_dataset('data/output/records.tf')
print("before")
print(dataset.take(1))
#%%
print(next(iter(dataset.take(1))))
#%%
