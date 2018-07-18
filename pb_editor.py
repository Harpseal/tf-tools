
#!/usr/bin/python3
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from google.protobuf import text_format
import sys
import flatbuffers
import os
# Add an op to initialize the variables.
#init_op = tf.global_variables_initializer()


#saver = tf.train.Saver()

#saved_pb_path = '/home/harpseal/dev/data/tf/proj/_models/pb/inception5h/tensorflow_inception_graph.pb'
saved_pb_path = '/home/harpseal/dev/data/tf/proj/_models/pb/inception_v3_2016_08_28_frozen/inception_v3_2016_08_28_frozen.pb'

if len(sys.argv) > 1:
    saved_pb_path = sys.argv[1]
#saved_pb_folder= '/home/harpseal/dev/data/tf/proj/_models/pb/inception5h/'


#with tf.Session() as sess:
#    saver.restore(sess, save_path)

# with tf.Session(graph=tf.Graph()) as sess:
#    tf.saved_model.loader.load(
#        sess,
#        [tf.saved_model.tag_constants.SERVING],
#        saved_pb_folder)

# Unpersists graph from file
if saved_pb_path.endswith(".pb"):
    print("tensorflow pb")
    with tf.gfile.GFile(saved_pb_path, "rb") as f:
        #proto_b=f.read()
        #print(proto_b)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tar_pad_nodes = {"FlowNetSD/Pad_14":"","FlowNetSD/Pad_16":"","FlowNetSD/Pad_18":"","FlowNetSD/Pad_20":""}
        for node in graph_def.node:
            if node.name in tar_pad_nodes.keys():
                for i in node.input:
                    if not i.startswith(node.name):
                        print(node.name + " " + str(i))
                        tar_pad_nodes[node.name] = i
        print(tar_pad_nodes)
        for node in graph_def.node:
            for idx in range(len(node.input)):
                if node.input[idx] in tar_pad_nodes.keys():
                    node.input[idx] = tar_pad_nodes[node.input[idx]]
                    print(node.name)
                    print(type(node.attr["padding"]))
                    #node.attr["padding"] = "SAME" tensorflow.core.framework.attr_value_pb2.AttrValue'>
                    node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=bytes("SAME",encoding = "utf8")))
                    # for a in range(len(node.attr)):
                    #     print(a)
                    #     print(node.attr["padding"])
                    #     print(type(node.attr[a].key))
                    #     if str(a.key) == "padding":
                    #         print(a.value)
                    print(node)
        #print(graph_def)
        tf.train.write_graph(graph_def, os.path.dirname(saved_pb_path),
                     os.path.basename(saved_pb_path)+'.out.pb', as_text=False)


if saved_pb_path.endswith(".tflite"):
    print("tensorflow lite")
    with open(saved_pb_path,'rb') as f:

        buf = bytearray(f.read())
        offset = 0
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        tab = flatbuffers.table.Table(buf,n + offset)
        print(tab)



    #text_format.Merge(proto_b, graph_def) 

#with tf.Session() as sess:
#    print(sess.graph.get_operations())