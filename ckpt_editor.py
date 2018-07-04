#!/usr/bin/python3
import tensorflow as tf
import os.path
from argparse import ArgumentParser

def show_ckpt(meta_path,ckpt_path):
    if not os.path.isfile(meta_path):
        print("Error: The meta {} doesn\'t exist".format(meta_path))
        return False
    if not os.path.isfile(ckpt_path):
        print("Error: The ckpt {} doesn\'t exist".format(ckpt_path))
        return False
    print("dirname {}\n  basename {}".format(os.path.dirname(ckpt_path),os.path.basename(ckpt_path)))
    tf.reset_default_graph()
    with tf.Session() as sess:
       saver = tf.train.import_meta_graph(meta_path)
       saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt_path),os.path.basename(ckpt_path)))
       #saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt_path)))
       sess.run(tf.global_variables_initializer())
       all_vars = tf.trainable_variables()
       for v in all_vars:
           print("%s with value %s" % (v.name, sess.run(v)))
    return True

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-ls", action='store_true', dest="mode_ls")
    parser.add_argument("-m", "--meta-path", dest="meta_path")
    parser.add_argument("-c", "--ckpt-path", dest="ckpt_path")
    

    args = parser.parse_args()
    print("meta_path {}  ckpt_path {}  mode_ls  {}".format(args.meta_path,args.ckpt_path,args.mode_ls))

    if args.mode_ls == True and args.meta_path != None and args.ckpt_path != None:
        res = show_ckpt(args.meta_path,args.ckpt_path)