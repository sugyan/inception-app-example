from flask import Flask, render_template, request
import argparse
import classify_image
import tensorflow as tf

# setup flags
classify_image.FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """Path to classify_image_graph_def.pb""")

# prepare to using trained model
classify_image.maybe_download_and_extract()
classify_image.create_graph()
node_lookup = classify_image.NodeLookup()
sess = tf.Session()
softmax_tensor = tf.squeeze(sess.graph.get_tensor_by_name('softmax:0'))

app = Flask(__name__)


@app.route('/recognize', methods=['POST'])
def recognize():
    f = request.files['image']
    predictions = sess.run(softmax_tensor,
                           feed_dict={'DecodeJpeg/contents:0': f.read()})
    results = []
    top_k = predictions.argsort()[-5:][::-1]
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        results.append({'label': human_string, 'score': score})
    return render_template('result.html', results=results)


@app.route('/')
def root():
    return render_template('index.html')
