from booknlp.booknlp import BookNLP

model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "custom",
    "entity_model_path": r"C:\Users\Adhyan\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0_modified.model",
    "coref_model_path": r"C:\Users\Adhyan\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0_modified.model",
    "quote_attribution_model_path": r"C:\Users\Adhyan\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1_modified.model",
    "bert_model_path": r"C:\Users\Adhyan\.cache\huggingface\hub"
}

booknlp = BookNLP("en", model_params)

# Input file (UTF-8 encoded)
input_file = "sample.txt"

# Output directory
output_directory = "output/"

# Book ID
book_id = "samples"

# Pass the filename, not the text content
booknlp.process(input_file, output_directory, book_id)
