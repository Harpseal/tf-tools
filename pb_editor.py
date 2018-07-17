
#!/usr/bin/python3
import tensorflow as tf
from google.protobuf import text_format
# Add an op to initialize the variables.
#init_op = tf.global_variables_initializer()


#saver = tf.train.Saver()

#saved_pb_path = '/home/harpseal/dev/data/tf/proj/_models/pb/inception5h/tensorflow_inception_graph.pb'
saved_pb_path = '/home/harpseal/dev/data/tf/proj/_models/pb/inception_v3_2016_08_28_frozen/inception_v3_2016_08_28_frozen.pb'
#saved_pb_folder= '/home/harpseal/dev/data/tf/proj/_models/pb/inception5h/'


#with tf.Session() as sess:
#    saver.restore(sess, save_path)

# with tf.Session(graph=tf.Graph()) as sess:
#    tf.saved_model.loader.load(
#        sess,
#        [tf.saved_model.tag_constants.SERVING],
#        saved_pb_folder)

# Unpersists graph from file
with tf.gfile.GFile(saved_pb_path, "rb") as f:
    #proto_b=f.read()
    #print(proto_b)
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    #for node in graph_def.node:
    #    print(node.name)
    print(graph_def)
    
    #text_format.Merge(proto_b, graph_def) 

#with tf.Session() as sess:
#    print(sess.graph.get_operations())