import os
import sys

def test_booknlp():
    try:
        print("Testing BookNLP import...")
        from booknlp.booknlp import BookNLP
        print("✓ BookNLP imported successfully")
        
        # Set up paths
        input_file = r"C:\Users\Adhyan\OneDrive\Desktop\EchoTales-Enhanced\data\raw\books\Lord Of The Mysteries.txt"
        output_directory = r"C:\Users\Adhyan\OneDrive\Desktop\EchoTales-Enhanced\modelsbooknlp\output"
        book_id = "lord_of_mysteries"
        
        # Verify input file exists
        if not os.path.exists(input_file):
            print(f"❌ Input file does not exist: {input_file}")
            return False
            
        print(f"✓ Input file found: {input_file}")
        
        # Verify output directory exists, create if not
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
            print(f"✓ Created output directory: {output_directory}")
        else:
            print(f"✓ Output directory exists: {output_directory}")
            
        # Model parameters
        model_params = {
            "pipeline": "entity,quote,supersense,event,coref",
            "model": "custom",
            "entity_model_path": r"C:\Users\Adhyan\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0_modified.model",
            "coref_model_path": r"C:\Users\Adhyan\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0_modified.model",
            "quote_attribution_model_path": r"C:\Users\Adhyan\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1_modified.model",
            "bert_model_path": r"C:\Users\Adhyan\.cache\huggingface\hub"
        }
        
        # Verify model files exist
        for key, path in model_params.items():
            if key.endswith('_path') and key != 'bert_model_path':
                if not os.path.exists(path):
                    print(f"❌ Model file does not exist: {path}")
                    return False
                else:
                    print(f"✓ Model file found: {os.path.basename(path)}")
        
        print("Initializing BookNLP...")
        booknlp = BookNLP("en", model_params)
        print("✓ BookNLP initialized successfully")
        
        print("Starting BookNLP processing...")
        print("This may take a while for a large book...")
        
        # Process the book
        booknlp.process(input_file, output_directory, book_id)
        
        print("✓ BookNLP processing completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure BookNLP and its dependencies are properly installed")
        return False
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("BookNLP Robust Test Script")
    print("=" * 50)
    success = test_booknlp()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed. Check the errors above.")
    sys.exit(0 if success else 1)