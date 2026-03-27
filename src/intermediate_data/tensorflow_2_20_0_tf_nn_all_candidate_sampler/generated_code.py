import tensorflow as tf

def call_func(inputs, num_true, num_sampled, unique, seed=0, name=None):
    sampled_candidates, true_expected_count, sampled_expected_count = tf.nn.all_candidate_sampler(
        true_classes=inputs,
        num_true=num_true,
        num_sampled=num_sampled,
        unique=unique,
        seed=seed,
        name=name
    )
    return [sampled_candidates, true_expected_count, sampled_expected_count]

batch_size = 2
num_true = 3
num_sampled = 5
true_classes = tf.random.uniform(shape=[batch_size, num_true], minval=0, maxval=num_sampled, dtype=tf.int64)
example_output = call_func(inputs=true_classes, num_true=num_true, num_sampled=num_sampled, unique=True)