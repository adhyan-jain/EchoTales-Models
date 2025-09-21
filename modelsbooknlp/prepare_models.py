import torch
import os

def remove_position_ids_and_save(model_file, device):
    if not os.path.isfile(model_file):
        print(f"File does not exist: {model_file}")
        return None
    
    state_dict = torch.load(model_file, map_location=device)
    modified = False
    if "bert.embeddings.position_ids" in state_dict:
        print(f"Removing 'position_ids' from {model_file}")
        del state_dict["bert.embeddings.position_ids"]
        modified = True
    save_path = model_file.replace(".model", "_modified.model")
    torch.save(state_dict, save_path)
    if modified:
        print(f"Saved modified model to {save_path}")
    else:
        print(f"No modification needed. Saved copy to {save_path}")
    return save_path


def process_model_files(model_params, device):
    updated_params = {}
    for key, path in model_params.items():
        if isinstance(path, str) and path.endswith(".model"):
            new_path = remove_position_ids_and_save(path, device)
            if new_path:
                updated_params[key] = new_path
            else:
                updated_params[key] = path
        else:
            updated_params[key] = path
    return updated_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "custom",
    "entity_model_path": r"C:\Users\Adhyan\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model",
    "coref_model_path": r"C:\Users\Adhyan\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model",
    "quote_attribution_model_path": r"C:\Users\Adhyan\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model",
    "bert_model_path": r"C:\Users\Adhyan\.cache\huggingface\hub"
}


    model_params = process_model_files(model_params, device)

    print("\nAll models processed. Use these paths in BookNLP:")
    for k, v in model_params.items():
        print(f"{k}: {v}")
