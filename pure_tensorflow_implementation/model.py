import tensorflow as tf
import tensorflow.contrib.slim.nets
import tensorflow.contrib.slim as slim
import tflearn
import inception_v4
import resnet_v2


resnet_101_weight_path = '/home/isaac/Desktop/canonical_model_weights/resnet101_v2/resnet_v2_101.ckpt'
vgg_weight_path = '/home/isaac/Desktop/canonical_model_weights/vgg16/vgg_16.ckpt'
reset_50_weight_path = '/home/isaac/Desktop/canonical_model_weights/resnet50_v2/resnet_v2_50.ckpt'
inception_v4_path = '/home/isaac/Desktop/canonical_model_weights/inception_v4/inception_v4.ckpt'
vgg_19_weight_path = '/home/isaac/Desktop/canonical_model_weights/vgg19/vgg_19.ckpt'
feature_extractor_path = vgg_weight_path

class Model:
    def __init__(self, tag = 'model1'):
        self.heightmap_ph = tf.placeholder(tf.float32, [None, 224, 224])
        self.extra_features_ph = tf.placeholder(tf.float32, [None, 22 + 35])
        self.labels_ph = tf.placeholder(tf.float32, [None, 2],
                                        name = 'labels_data_placeholder')
        self.keep_prob_ph = tf.placeholder(tf.float32, [],
                                        name = 'keep_prob_placeholder')
        self.sess = tf.Session()

        preprocessed_images, preprocessed_extra_features =\
            self.preprocess(self.heightmap_ph, self.extra_features_ph)

        temp = set(tf.global_variables())

        # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        # print_tensors_in_checkpoint_file(feature_extractor_path, '', False)
        # quit()


        # ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/pool1', 'vgg_16/conv2/conv2_1',
        # 'vgg_16/conv2/conv2_2', 'vgg_16/pool2', 'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2',
        # 'vgg_16/conv3/conv3_3', 'vgg_16/pool3', 'vgg_16/conv4/conv4_1', 'vgg_16/conv4/conv4_2',
        # 'vgg_16/conv4/conv4_3', 'vgg_16/pool4', 'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2',
        # 'vgg_16/conv5/conv5_3', 'vgg_16/pool5', 'vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8']

        self.feature_extractor = tf.contrib.slim.nets.vgg.vgg_16(preprocessed_images)[1]['vgg_16/pool5']
        # self.feature_extractor = tf.contrib.slim.nets.vgg.vgg_19(preprocessed_images)[1]['vgg_19/conv5/conv5_4']
        # self.feature_extractor = tf.contrib.slim.nets.resnet_v2.resnet_v2_101(preprocessed_images)[1]['resnet_v2_101/block4']

        # self.feature_extractor = tf.contrib.slim.nets.inception.inception_v4(preprocessed_images)[1]['Mixed_7d']
        # self.feature_extractor = resnet_v2.resnet_v2_50(preprocessed_images)[1]['resnet_v2_50/block4']

        self.feature_extractor_vars = set(tf.global_variables()) - temp
        self.feature_extractor_saver = tf.train.Saver(self.feature_extractor_vars)
        self.feature_extractor_saver.restore(self.sess, feature_extractor_path)

        self.predicted_logits = self.predict_logits(self.feature_extractor, preprocessed_extra_features)


        self.pred_probabilities = tf.nn.softmax(self.predicted_logits)
        crossentropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
            logits=self.predicted_logits,
            labels=self.labels_ph,
            name="crossentropy"))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predicted_logits, 1),
                    tf.argmax(self.labels_ph,1)), tf.float32))
        self.loss = crossentropy
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("y_labels", self.labels_ph)
        tf.summary.histogram("predicted_probabilities", self.pred_probabilities)

        added_layer_variables = list(set(tf.global_variables()) - self.feature_extractor_vars)

        temp = set(tf.global_variables())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=.002)\
                .minimize(self.loss, var_list=added_layer_variables)
        self.optimizer_variables = set(tf.global_variables()) - temp

        init = tf.initialize_variables(list(set(added_layer_variables).union(self.optimizer_variables)))
        self.saver = tf.train.Saver(tf.all_variables())
        self.sess.run(init)
        self.merged_summaries = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter('./logdir' + '/' + tag, self.sess.graph)


    def save(self, global_step):
        self.saver.save(self.sess, 'checkpoints/model1', global_step=global_step)

    def restore(self):
        # self.saver.save(self.sess, 'model1', global_step=global_step)
        latest_checkpoint = tf.train.latest_checkpoint("/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation/checkpoints")
        self.saver.restore(self.sess, latest_checkpoint)

    def preprocess(self, heightmap_batch, extra_features):
        extra_features = tf.log(extra_features + .001)
        column_mins = tf.reduce_min(extra_features, axis=0)
        extra_features -= column_mins
        column_maxes = tf.reduce_max(extra_features, axis=0)
        extra_features /= column_maxes + .001

        heightmap_batch = heightmap_batch -  tf.reshape(tf.reduce_min(heightmap_batch, axis=[1,2]), [-1, 1, 1])
        heightmap_batch = heightmap_batch  / tf.reshape(tf.reduce_max(heightmap_batch, axis=[1,2]), [-1, 1, 1])
        heightmap_batch = heightmap_batch * 255
        heightmap_image_batch = tf.stack([heightmap_batch] * 3, axis=-1)
        channel_means = [[[103.939, 116.779, 123.68]]]
        heightmap_image_batch -= channel_means
        tf.summary.image("heightmaps", heightmap_image_batch)
        self.heightmap_image_batch = heightmap_image_batch
        return heightmap_image_batch, extra_features

    def predict_logits(self, feature_extractor, preprocessed_extra_features):
        print "feature extractor shape", feature_extractor.get_shape()
        network = tflearn.layers.conv.conv_2d(feature_extractor,
                                              64,
                                              3, strides = 1,
                                              activation = 'leakyrelu',
                                              name = 'tflearn_conv_layer')

        network = tf.nn.dropout(network, self.keep_prob_ph)
        network = tflearn.layers.fully_connected(network, 32, activation = 'leakyrelu',
                                                 name = 'tflearn_layer1')
        network = tf.nn.dropout(network, self.keep_prob_ph)
        network = tf.concat([network, preprocessed_extra_features], 1)
        network = tflearn.layers.fully_connected(network, 32, activation = 'leakyrelu',
                                                 name = 'tflearn_layer3')
        network = tf.nn.dropout(network, self.keep_prob_ph)

        network = tflearn.layers.fully_connected(network, 2, activation = 'linear',
                                                 name = 'tflearn_layer4')
        return network
