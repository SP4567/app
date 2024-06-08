import streamlit as st
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load BERT model and tokenizer
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    return tokenizer, model


# Function to get embeddings for each word
def get_word_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # tuple of (layer, batch_size, seq_length, hidden_size)
    word_embeddings = torch.stack(hidden_states, dim=0).squeeze(1).detach().numpy()
    return word_embeddings


# Function to apply PCA
@st.cache_resource
def apply_pca(embeddings, n_components):
    n_samples, n_features = embeddings.shape
    st.write(f"Number of samples: {n_samples}, Number of features: {n_features}")

    if n_components > min(n_samples, n_features):
        n_components = min(n_samples, n_features)
        st.write(f"Adjusted n_components to: {n_components}")

    pca = PCA(n_components=n_components)
    transformed_embeddings = pca.fit_transform(embeddings)

    st.write(f"Transformed embeddings shape: {transformed_embeddings.shape}")
    return transformed_embeddings


# Streamlit app
def main():
    st.title("Text Embedding and Visualization")

    # Text input
    text_input = st.text_area("Enter text (separate sentences with new lines):")

    if st.button("Generate Embeddings"):
        if text_input:
            texts = text_input.split('\n')
            st.write(f"Input texts: {texts}")

            tokenizer, model = load_bert()

            for text in texts:
                st.write(f"Text: {text}")
                word_embeddings = get_word_embeddings(text, tokenizer, model)
                n_layers, seq_length, hidden_size = word_embeddings.shape
                st.write(f"Embeddings shape (layers x tokens x features): {word_embeddings.shape}")

                for layer in range(n_layers):
                    embeddings = word_embeddings[layer]

                    # Apply PCA to reduce to 2D
                    embeddings_2d = apply_pca(embeddings, 2)
                    # Apply PCA to reduce to 3D
                    embeddings_3d = apply_pca(embeddings, 3)

                    st.write(f"Layer {layer + 1}")

                    # Plot 2D embeddings
                    fig, ax = plt.subplots()
                    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
                    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text, return_tensors='pt')['input_ids'][0])
                    for i, token in enumerate(tokens):
                        ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], token)
                    st.pyplot(fig)

                    # Plot 3D embeddings
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2])
                    for i, token in enumerate(tokens):
                        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], token)
                    st.pyplot(fig)

        else:
            st.error("Please enter some text.")


if __name__ == "__main__":
    main()
