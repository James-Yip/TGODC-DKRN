import tensorflow as tf

if __name__ == '__main__':
    with tf.Graph().as_default():
        a = tf.placeholder(tf.int32)
        matrix = tf.ones(shape=[5,5])
        embed = tf.random_normal(shape=[5,5])
        # new_embed = tf.map_fn(lambda x: tf.sparse_to_dense(x, [5], 1., 0., False), a, dtype=tf.float32, parallel_iterations=True)
        new_embed = tf.sparse_to_dense(a, [5], 1., 0., False)
        new_embed = embed * tf.tile(tf.transpose(tf.expand_dims(new_embed,axis=0)),[1,5])

        matrix1 = tf.ones(shape=[64, 2678,200])
        matrix2 = tf.ones(shape=[200,2678])
        # matrix3 = tf.matmul(matrix1, matrix2)
        matrix3 = tf.scan(lambda a, b: tf.matmul(b,matrix2), matrix1,initializer=tf.zeros([2678,2678]))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(matrix))
            print(sess.run(embed))
            print(sess.run(new_embed,feed_dict={a:(1,3)}))
            print(sess.run(matrix3))