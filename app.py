from scipy.spatial import distance
from jinja2 import TemplateNotFound

from sentence_transformers import SentenceTransformer, util
import spacy
from itertools import combinations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import transformers
import numpy as np
import codecs
import tensorflow as tf
from typing import Dict, List, Union, Optional
import ast
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import tensorflow_datasets as tfds
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Progbar
import spacy
import nltk
from nltk.corpus import stopwords

import re

nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
command = [
    {"id": 1, "name": "Increase font size"},
    {"id": 2, "name": "Decrease font size"},
    {"id": 3, "name": "Simplify webpage"},
    {"id": 4, "name": "Change font"},
    {"id": 5, "name": "Increase brightness"},
    {"id": 6, "name": "Decrease brightness"},
    {"id": 7, "name": "Navigate to home page"},
    {"id": 8, "name": "Navigate to about page"},
    {"id": 9, "name": "Navigate to navigation page"},
    {"id": 10, "name": "Navigate to graph page"},
    {"id": 11, "name": "Navigate to log in page"},
    {"id": 12, "name": "Navigate to sign up page"},
]

commandNavigation = [
    {"id": 1, "name": "Navigate to home page"},
    {"id": 2, "name": "Navigate to search page"},
    {"id": 3, "name": "Navigate to profile page"},
    {"id": 4, "name": "Navigate to activity page"},
    {"id": 5, "name": "Create a post"},
    {"id": 6, "name": "View a story"},
    {"id": 7, "name": "Like a post"},
    {"id": 8, "name": "Comment on a post"},
    {"id": 9, "name": "Share a post"},
    {"id": 10, "name": "Save a post"},
    {"id": 11, "name": "Follow a user"},
    {"id": 12, "name": "Send a message"},
    {"id": 13, "name": "Open settings"},
]
model = SentenceTransformer("all-MiniLM-L6-v2")


class TimeSeriesObj(BaseModel):
    time: Optional[str] = None
    positionX: Optional[float] = None
    positionY: Optional[float] = None
    eyeX: Optional[float] = None
    eyeY: Optional[float] = None
    hoverType: Optional[str] = None
    isMouseDown: Optional[bool] = None
    scrollDirection: Optional[str] = None


class TimeSeries(BaseModel):
    keys: List[TimeSeriesObj]


class Message(BaseModel):
    message: str


class Feature(BaseModel):
    type: str
    uniqueValues: List[Union[str, float, int]]


class Filter(BaseModel):
    message: str
    keys: Dict[str, Feature]


num_layers = 4
d_model = 128
dff = 256
num_heads = 4
input_vocab_size = (
    tfds.deprecated.text.SubwordTextEncoder.load_from_file("./tokenizer_q").vocab_size
    + 2
)
target_vocab_size = (
    tfds.deprecated.text.SubwordTextEncoder.load_from_file("./tokenizer_a").vocab_size
    + 2
)
dropout_rate = 0.1
tokenizer_a = tfds.deprecated.text.SubwordTextEncoder.load_from_file("./tokenizer_a")
tokenizer_q = tfds.deprecated.text.SubwordTextEncoder.load_from_file("./tokenizer_q")


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, :]

    return tf.cast(pos_encoding, dtype=tf.float32)


pos_encoding = positional_encoding(50, 512)


def create_padding_mask(seq):
    """
    seq: padded sentence length (5)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # Adding 2, 3 dimn using tf.newaxis, 2-> As this mask will be multiplied with each attention head and 3-> for each word in a sentance
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    The look-ahead mask is used to mask the future tokens in a sequence
    """
    # band_part with this setting creates lower triangular matrix that's why subtracting from 1
    # [[0., 1., 1.],
    #  [0., 0., 1.],
    #  [0., 0., 0.]] output with size:3
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
    q: query shape == (..., seq_len_q, depth) # NOTE: depth=dk
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk. underroot d_model i.e. underroot(100)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (
            mask * -1e9
        )  # -1e9 ~ (-INFINITY) => where ever mask is set, make its logit value close to -INF
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model  # typically 512

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):  # dff = 512
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(
            x + attn_output
        )  # (batch_size, input_seq_len, d_model) # with Attention

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model) #with Attention

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out2
        )  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.5,
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        seq_len = tf.shape(x)[1]  # x:(batch, seq_len)
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.5,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(
        self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None
    ):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x,
                enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        rate=0.5,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
        )

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self,
        inp,
        tar,
        training=False,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    ):
        enc_output = self.encoder(
            inp, training=training, mask=enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            x=tar,
            enc_output=enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask,
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

temp_learning_rate_schedule = CustomSchedule(d_model)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = class_transformer(
            inp,
            tar_inp,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask,
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, class_transformer.trainable_variables)

    optimizer.apply_gradients(zip(gradients, class_transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


class_transformer = Transformer(
    num_layers,
    d_model,
    num_heads,
    dff,
    input_vocab_size,
    target_vocab_size,
    pe_input=input_vocab_size,
    pe_target=target_vocab_size,
    rate=dropout_rate,
)

temp_input = tf.random.uniform((128, 40), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((128, 40), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = class_transformer(
    temp_input,
    temp_target,
    training=False,
    enc_padding_mask=None,
    look_ahead_mask=None,
    dec_padding_mask=None,
)

fn_out.shape
class_transformer.load_weights("./transformer_weight.h5")
class_transformer.summary()


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI app!"}


MAX_LENGTH = 40


def evaluate_r(inp_sentence, temperature=0.0):
    start_token = [tokenizer_q.vocab_size]
    end_token = [tokenizer_q.vocab_size + 1]

    inp_sentence = start_token + tokenizer_q.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_a.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output
        )

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = class_transformer(
            encoder_input,
            output,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask,
        )

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        # Adjust the predictions with the temperature
        predictions = predictions / temperature

        # Sample from the distribution instead of taking the max
        predicted_id = tf.random.categorical(predictions[:, 0, :], num_samples=1)[0, 0]

        predicted_id = tf.cast(predicted_id, tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_a.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # expand dimensions of predicted_id to match the shape of output
        predicted_id = tf.expand_dims([predicted_id], 0)

        # concatenate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


@app.post("/send-message")
async def send_message(message: Message):

    test_vec = model.encode([message.message])[0]
    similarity_arr = []

    for sent in command:

        similarity_score = 1 - distance.cosine(
            test_vec, model.encode([sent["name"]])[0]
        )
        similarity_score = float(similarity_score)

        if similarity_score > 0.01:
            similarity_arr.append(
                {"score": similarity_score, "text": sent["name"], "id": sent["id"]}
            )

    print(similarity_arr)
    similarity_arr = sorted(similarity_arr, key=lambda x: x["score"], reverse=True)
    print(similarity_arr)
    return similarity_arr


@app.post("/send-message-navigation")
async def send_message(message: Message):
    # print(message.message)
    test_vec = model.encode([message.message])[0]
    similarity_arr = []

    for sent in commandNavigation:
        print(sent)

        # Calculate similarity score
        similarity_score = 1 - distance.cosine(
            test_vec, model.encode([sent["name"]])[0]
        )

        # Convert the numpy.float32 to Python float
        similarity_score = float(similarity_score)

        if similarity_score > 0.01:
            similarity_arr.append(
                {"score": similarity_score, "text": sent["name"], "id": sent["id"]}
            )

    print(similarity_arr)
    similarity_arr = sorted(similarity_arr, key=lambda x: x["score"], reverse=True)
    print(similarity_arr)
    return similarity_arr


@app.post("/predict-message")
async def send_message(message: Message):

    sentence_o = ""
    a, b = evaluate_r(message.message, 0.01)
    for i in a[1:]:
        print(tokenizer_a.decode([i]), end="")
        sentence_o += "" + tokenizer_a.decode([i])
    sentence = sentence_o
    print(sentence)
    return sentence


# --- Helper Functions ---


def generate_ngrams(tokens, min_n=2, max_n=6):
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


def extract_phrases(target_phrases, sentence, threshold=0.5):
    """Extract overlapping n-grams from the sentence that match target phrases."""
    doc = nlp(sentence)
    sentence_tokens = [token.text for token in doc]
    ngrams = generate_ngrams(sentence_tokens, min_n=2, max_n=6)
    if not ngrams:
        return []
    print("Target phrases:", target_phrases)
    target_embeddings = model.encode(target_phrases)
    ngram_embeddings = model.encode(ngrams)
    cos_sim = util.cos_sim(target_embeddings, ngram_embeddings)
    matches = []
    for i, phrase in enumerate(target_phrases):
        for j, ng in enumerate(ngrams):
            if cos_sim[i][j] > threshold:
                matches.append((ng, cos_sim[i][j].item(), i))
    matches.sort(key=lambda x: (-x[1], -len(x[0].split())))
    used_indices = set()
    selected = []

    def get_span(ngram):
        words = ngram.split()
        for i in range(len(sentence_tokens) - len(words) + 1):
            if sentence_tokens[i : i + len(words)] == words:
                return (i, i + len(words) - 1)
        return None

    for match in matches:
        ng = match[0]
        idx = match[2]
        span = get_span(ng)
        if not span:
            continue
        start, end = span
        overlap = any(not (end < s or start > e) for s, e in used_indices)
        if not overlap:
            selected.append((target_phrases[idx], ng))
            used_indices.add((start, end))
    selected.sort(key=lambda pair: get_span(pair[1])[0])
    return selected


from num2words import num2words


def process_sentence_with_placeholders(sentence, matched_pairs, target_phrases):
    """Replace each occurrence of a matched target phrase with a placeholder."""
    placeholder_sentence = sentence
    replacements = {}
    for original_phrase, matched_text in matched_pairs:
        idx = target_phrases.index(original_phrase)
        placeholder = f"<target {idx + 1}>"
        placeholder_sentence = placeholder_sentence.replace(
            matched_text, placeholder, 1
        )
        replacements[placeholder] = matched_text
    return placeholder_sentence, replacements


def convert_word_to_num(word):
    word = word.lower()
    mapping = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    try:
        return float(word)
    except ValueError:
        return mapping.get(word, None)


# --- Modified process_clause ---
def process_clause(clause, target_index, target_phrases, matched_indexes, keys):
    """
    Process a clause. For numeric comparisons (between, more than, less than), handle as before.
    For a clause with "to be", match against the actual attribute values, and use the matched index if available.
    """
    output_clause = clause
    target_entries = []
    # Determine the attribute name using target_index (1-based)
    attr_index = int(target_index) - 1
    attribute_name = (
        target_phrases[attr_index] if attr_index < len(target_phrases) else None
    )
    conjunction_phrases = ["between", "lesser", "greater"]
    matched_pairs = list(extract_phrases(conjunction_phrases, clause, 0.6))
    if len(matched_pairs) == 0:
        feature = keys[target_phrases[int(target_index) - 1]]
        target_word = feature.uniqueValues[
            matched_indexes[target_phrases[int(target_index) - 1]][0] - 1
        ].lower()
        pattern = r"(\b" + re.escape(target_word) + r"\b)"
        match = re.search(pattern, clause.lower())
        if match:
            value_text = match.group(1)
            placeholder_label = (
                f"<label { matched_indexes[target_phrases[int(target_index)-1]][0]}>"
            )
            output_clause = clause.replace(value_text, placeholder_label)
            target_entries.append(
                {
                    f"label {matched_indexes[target_phrases[int(target_index)-1]][0]}": value_text
                }
            )
    elif "between" in list(matched_pairs[0])[1]:
        digit_count = 0
        num1 = None
        num2 = None
        target_entries = []

        clause_words = clause.split()
        for i, word in enumerate(clause_words):
            if word.isdigit():
                digit_count += 1
                if digit_count == 1:
                    num1 = word
                    clause_words[i] = f"<num {target_index} sm>"
                elif digit_count == 2:
                    num2 = word
                    clause_words[i] = f"<num {target_index} lg>"
                    break
        output_clause = " ".join(clause_words)

        # Append the target entries
        target_entries.append({f"num {num2words(target_index)} sm": num1})
        target_entries.append({f"num {num2words(target_index)} lg": num2})

    elif "more than" in list(matched_pairs[0])[1]:
        digit_count = 0
        num1 = None
        target_entries = []

        clause_words = clause.split()
        for i, word in enumerate(clause_words):
            if word.isdigit():
                num1 = word
                clause_words[i] = f"<num {target_index} sm>"
                break
        output_clause = " ".join(clause_words)

        target_entries.append({f"num {num2words(target_index)} sm": num1})

    elif "less than" in list(matched_pairs[0])[1]:
        digit_count = 0
        num1 = None
        target_entries = []

        clause_words = clause.split()
        for i, word in enumerate(clause_words):
            if word.isdigit():
                num1 = word
                clause_words[i] = f"<num {target_index} sm>"
                break
        output_clause = " ".join(clause_words)
        target_entries.append({f"num {num2words(target_index)} sm": num1})

    return output_clause, target_entries


# --- Modified process_sentence ---
def process_sentence(
    sentence, preprocessed_sentence, data_dict, matched_indexes, target_phrases, keys
):
    """
    Process each clause in the preprocessed sentence, replacing target phrases with
    appropriate placeholders and handling numeric/categorical comparisons.
    """
    clauses = [cl.strip() for cl in preprocessed_sentence.split(",")]
    modified_clauses = []
    all_target_entries = []
    for clause in clauses:
        target_match = re.search(r"<target (\w+)>", clause)
        if target_match:
            target_index = target_match.group(1)
            modified_clause, target_entries = process_clause(
                clause, target_index, target_phrases, matched_indexes, keys
            )
            modified_clauses.append(modified_clause)
            all_target_entries.extend(target_entries)
        else:
            modified_clauses.append(clause)

    modified_sentence = ", ".join(modified_clauses)
    return modified_sentence, all_target_entries


# --- FastAPI Endpoint using the Pydantic Filter model ---
@app.post("/filter_sentence")
async def send_message(filter: Filter):

    filtered_sentence = " ".join(
        [
            word
            for word in filter.message.split()
            if word.lower() not in stopwords.words("english")
        ]
    )
    print("Filtered sentence:", filtered_sentence)
    doc = nlp(filtered_sentence)

    number_info = []
    for i, token in enumerate(doc):
        if token.like_num:
            try:
                num_value = float(token.text)
            except ValueError:
                continue
            closest = None
            for j in range(i - 1, -1, -1):
                if doc[j].pos_ in ["NOUN", "PROPN", "ADJ"]:
                    if j > 0 and doc[j - 1].dep_ == "compound":
                        closest = f"{doc[j - 1].text} {doc[j].text}"
                    else:
                        closest = doc[j].text
                    break
            number_info.append((num_value, closest))
    number_info.sort(key=lambda x: x[0])
    result_dict = {num: label for num, label in number_info}
    print("Extracted number info:", result_dict)

    matched_indexes_dict = {}
    for attribute, info in filter.keys.items():

        if info.type == "number" and ("min" not in info or "max" not in info):
            try:
                numeric_values = [float(x) for x in info.uniqueValues]
                info.__dict__.update(
                    {"min": min(numeric_values), "max": max(numeric_values)}
                )
            except Exception as e:
                print(f"Error processing attribute '{attribute}': {e}")
        matched_indexes = []
        attribute_tokens = attribute.lower().split()
        for token in doc:
            if token.text.lower() == attribute_tokens[0]:
                if all(
                    (token.i + i < len(doc))
                    and (doc[token.i + i].text.lower() == attribute_tokens[i])
                    for i in range(len(attribute_tokens))
                ):
                    window_size = 8
                    start = max(token.i - window_size, 0)
                    end = min(token.i + window_size, len(doc))
                    for nearby_token in doc[start:end]:
                        nearby_word = nearby_token.text.lower()
                        if info.type == "string" and nearby_word in [
                            v.lower() for v in info.uniqueValues
                        ]:
                            matched_indexes.append(
                                [v.lower() for v in info.uniqueValues].index(
                                    nearby_word
                                )
                                + 1
                            )
                            break
                        elif info.type == "number" and nearby_token.like_num:
                            try:
                                value = float(nearby_token.text)
                                if (
                                    info.__dict__.get("min")
                                    <= value
                                    <= info.__dict__.get("max")
                                ):
                                    matched_indexes.append(value)
                                    break
                            except ValueError:
                                continue
        if matched_indexes:
            matched_indexes_dict[attribute] = matched_indexes
    print("Matched indexes for attributes:", matched_indexes_dict)

    target_phrases = list(filter.keys.keys())
    matched_pairs = extract_phrases(target_phrases, filter.message)
    print("Matched target phrase pairs:", matched_pairs)
    processed_sentence, replacements = process_sentence_with_placeholders(
        filter.message, matched_pairs, target_phrases
    )
    print("Processed sentence with placeholders:", processed_sentence)
    print("Replacements mapping:", replacements)

    # Step 4: Final processing of the sentence.
    modified_sentence, target_dict = process_sentence(
        filter.message,
        processed_sentence,
        result_dict,
        matched_indexes_dict,
        target_phrases,
        filter.keys,
    )

    if len(replacements) != 0:
        words = modified_sentence.split()

        for index, word in enumerate(words):

            if word.isdigit():
                words[index] = num2words(int(word))

            else:

                new_word = ""
                for char in word:

                    if char.isdigit():
                        new_word += num2words(int(char))
                    else:
                        new_word += char
                words[index] = new_word

        modified_sentence = " ".join(words)
        print("Final modified sentence:", modified_sentence)
        print("Final target dictionary:", target_dict)
        sentence_o = ""
        a, b = evaluate_r(modified_sentence, 0.01)
        for i in a[1:]:
            print(tokenizer_a.decode([i]), end="")
            sentence_o += "" + tokenizer_a.decode([i])
        sentence = sentence_o
        print(sentence)
        return {"sentence": sentence, "target_string": target_dict}
    else:
        return {}


import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam

# Define CNN for positionX, positionY, eyeX, eyeY
cnn_input = Input(shape=(250, 4))  # 4 features: positionX, positionY, eyeX, eyeY

cnn_layer1 = Conv1D(128, 3, activation="relu")(cnn_input)
cnn_layer2 = MaxPooling1D(2)(cnn_layer1)
cnn_layer3 = Dropout(0.2)(cnn_layer2)

cnn_layer4 = Conv1D(256, 3, activation="relu")(cnn_layer3)
cnn_layer5 = MaxPooling1D(2)(cnn_layer4)
cnn_layer6 = Dropout(0.2)(cnn_layer5)

cnn_layer7 = Conv1D(256, 3, activation="relu")(cnn_layer6)
cnn_layer8 = MaxPooling1D(2)(cnn_layer7)
cnn_layer9 = Dropout(0.2)(cnn_layer8)

cnn_flattened = Flatten()(cnn_layer9)

# Define the fully connected network for other features: hoverType, isMouseDown, scrollDirection
dense_input = Input(shape=(250, 3))  # Assuming X_train_dense has the remaining features

dense_layer1 = Dense(1000, activation="relu")(dense_input)
dense_layer2 = Dropout(0.2)(dense_layer1)

dense_layer3 = Dense(2560, activation="relu")(dense_layer2)
dense_layer4 = Dropout(0.2)(dense_layer3)

# Flatten the dense part to match the CNN output for concatenation
dense_flattened = Flatten()(dense_layer4)

# Merge CNN and Dense branches
merged = Concatenate()([cnn_flattened, dense_flattened])

# Final output layer for classification (5 classes)
output = Dense(3, activation="softmax")(merged)

# Define the model
cnn_model = Model(inputs=[cnn_input, dense_input], outputs=output)


@app.post("/get-issue")
async def send_message(timeSeries: TimeSeries):
    df = pd.DataFrame([dict(row) for row in timeSeries.keys])

    import pickle

    # Load from file
    with open(r"C:\Users\Asus\Downloads\UNCLE\uncle\src\label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print(df)
    df["hoverType"] = label_encoder.fit_transform(df["hoverType"])
    df["scrollDirection"] = label_encoder.fit_transform(df["scrollDirection"])
    df["isMouseDown"] = df["isMouseDown"].astype(int)
    df[["eyeX", "eyeY"]] = df[["eyeX", "eyeY"]].fillna(-1)
    # Reorder the columns
    new_column_order = [
        "positionX",
        "positionY",
        "eyeX",
        "eyeY",
        "hoverType",
        "isMouseDown",
        "time",
        "scrollDirection",
    ]
    df = df[new_column_order]
    sampled_data = np.array(
        df[
            [
                "positionX",
                "positionY",
                "eyeX",
                "eyeY",
                "hoverType",
                "isMouseDown",
                "scrollDirection",
            ]
        ]
    )

    cnn_model.load_weights(r"C:\Users\Asus\Downloads\UNCLE\uncle\real.h5")
    sampled_data = (sampled_data - sampled_data.min()) / (
        sampled_data.max() - sampled_data.min()
    )
    cnn = sampled_data[:, :4]  # First 4 columns
    dense = sampled_data[:, 4:]
    result = np.argmax(cnn_model.predict([np.array([cnn]), np.array([dense])]))
    print(result)
    print(result.dtype)
    return result.item()
